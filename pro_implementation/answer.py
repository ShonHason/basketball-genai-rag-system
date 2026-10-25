import os
import threading
import time
from itertools import zip_longest
from openai import OpenAI
from dotenv import load_dotenv
from chromadb import PersistentClient
from litellm import completion
from pydantic import BaseModel, Field
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv(override=True)

# --- Configuration ---
MODEL = "openai/gpt-5-nano"
# GPT-5 models "think" before answering; low keeps cost and latency down
REASONING_EFFORT = "low"
# "local" = free cross-encoder (BAAI/bge-reranker-v2-m3) running on this machine, "llm" = MODEL as reranker
RERANKER = "local"
LOCAL_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
import litellm
litellm.suppress_debug_info = True
litellm.set_verbose = False
# GPT-5 family only accepts the default temperature; drop unsupported params instead of failing
litellm.drop_params = True
# litellm's default timeout is 6000s, so one hung request could stall a whole evaluation run
litellm.request_timeout = 120
#MODEL = "openrouter/openai/gpt-oss-120b" 
DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
collection_name = "docs"
embedding_model = "text-embedding-3-large"
#wait = wait_exponential(multiplier=1, min=10, max=240)
wait = wait_exponential(multiplier=2, min=10, max=60)
# Without a stop condition tenacity retries forever (e.g. on a bad API key)
stop = stop_after_attempt(4)
# Initialize Chroma and OpenAI clients
openai = OpenAI(timeout=60)
chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(collection_name)
DEBUG = True


def use_collection(name: str):
    """Point retrieval at another collection (e.g. a different chunk size built by ingest.py)."""
    global collection, collection_name
    collection = chroma.get_collection(name)
    collection_name = name

RETRIEVAL_K = 15
FINAL_K = 7


def model_kwargs() -> dict:
    """Extra completion params for the generator model."""
    # Explicit per-call timeout: the global litellm.request_timeout is not applied on every provider route
    kwargs = {"timeout": 120}
    if "gpt-5" in MODEL:
        kwargs["reasoning_effort"] = REASONING_EFFORT
    return kwargs


# --- Local cross-encoder reranker -------------------------------------------
# One model shared by all threads; predictions run one at a time on the GPU.
_reranker = None
_reranker_lock = threading.Lock()
# Time each thread spent queuing for the reranker, so evaluation latency
# measures the pipeline and not contention between parallel test workers.
_wait = threading.local()


def reranker_wait_seconds() -> float:
    return getattr(_wait, "seconds", 0.0)


def reset_reranker_wait():
    _wait.seconds = 0.0


def _free_gpu_cache():
    """Release PyTorch's cached GPU memory so it doesn't accumulate across calls."""
    import torch

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def rerank_local(question, chunks):
    """Score every (question, chunk) pair with a cross-encoder and sort by score."""
    global _reranker
    if not chunks:
        return []
    queued = time.perf_counter()
    with _reranker_lock:
        _wait.seconds = reranker_wait_seconds() + time.perf_counter() - queued
        if _reranker is None:
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            import torch
            from sentence_transformers import CrossEncoder

            # fp16 halves the weights' memory; on a 16GB Mac the fp32 model plus PyTorch's
            # GPU cache grew to ~15GB and pushed the system into swap
            kwargs = {"model_kwargs": {"dtype": torch.float16}} if torch.backends.mps.is_available() else {}
            _reranker = CrossEncoder(LOCAL_RERANKER_MODEL, max_length=1024, **kwargs)
        scores = _reranker.predict([(question, chunk.page_content) for chunk in chunks], batch_size=8)
        _free_gpu_cache()
    order = sorted(range(len(chunks)), key=lambda i: -scores[i])
    return [chunks[i] for i in order]

SYSTEM_PROMPT = """
You are a professional sports analyst for the Hoopball League.
Use the provided context to answer questions about rosters, stats, and history.

CRITICAL INSTRUCTIONS:
1. Answer ONLY based on the provided extracts. 
2. If the answer is not in the context, say: "I'm sorry, I don't have that information in the league records."
3. Be extremely precise with player names and stats.
4. If you refer to previous parts of the conversation, ensure it aligns with the provided context.

Context:
{context}
"""

class Result(BaseModel):
    page_content: str
    metadata: dict

class RankOrder(BaseModel):
    order: list[int] = Field(description="Relevance order of chunks.")

def clean_gradio_history(history):
    """
    Converts Gradio 6.0 multimodal dictionary history into clean text 
    that the LLM and Rewriter can process.
    """
    clean = []
    for msg in history:
        content = msg["content"]
        if isinstance(content, list):
            # Extract text from [{'text': '...', 'type': 'text'}]
            text = next((item["text"] for item in content if item["type"] == "text"), "")
        else:
            text = content
        clean.append({"role": msg["role"], "content": text})
    return clean

@retry(wait=wait, stop=stop)
def rerank(question, chunks):
    """
    Advanced Reranker: Acts as an aggressive filter to remove noise from K=50 retrieval.
    Focuses on Entity matching, Data Density (tables), and Source relevance.
    """
    if not chunks: 
        return []
    
    # System prompt designed to force the LLM to prioritize hard facts over summaries
    system_prompt = """
    You are a High-Precision Sports Reranker. 
    Rank chunks by their ability to provide the SPECIFIC answer.

    STRICT RANKING HIERARCHY:
    1. RAW DATA: Any chunk containing a Markdown Table with the relevant stats MUST be Rank #1.
    2. CONFLICT RESOLUTION: If stats in a table contradict a text summary, prioritize the TABLE.
    3. ENTITY MATCH: Chunks from the specific player's .md file take priority over league-wide summaries.
    4. COMPARISON DIVERSITY: If comparing two players, ensure chunks for BOTH are in the top 4.

    Return ONLY the RankOrder JSON: {"order": [ID1, ID2, ...]}
    """

    user_prompt = f"Question: {question}\n\n"
    for i, chunk in enumerate(chunks):
        # Extracting the filename from metadata helps the LLM recognize player-specific files
        source = chunk.metadata.get('source', 'unknown')
        content_preview = chunk.page_content[:200].replace('\n', ' ')
        user_prompt += f"### ID {i+1} | Source: {source} | Preview: {content_preview}...\nFull Content:\n{chunk.page_content}\n\n"
    try:
        response = completion(
            model=MODEL, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=RankOrder,
            # temperature=0 is critical for consistency and prevents "hallucinated" rankings
            temperature=0,
            **model_kwargs(),
        )
        
        reply = response.choices[0].message.content
        order = RankOrder.model_validate_json(reply).order
        
        # Build the ranked list using the original Result objects
        reranked = [chunks[i - 1] for i in order if 0 < i <= len(chunks)]
        
        # Fallback: If the LLM didn't rank all chunks, append the missing ones at the end
        # This ensures we don't lose potential evidence if the LLM was lazy
        seen_indices = {i-1 for i in order if 0 < i <= len(chunks)}
        for i in range(len(chunks)):
            if i not in seen_indices:
                reranked.append(chunks[i])
                
        return reranked

    except Exception as e:
        print(f"[DEBUG] Rerank parsing failed: {e}")
        return chunks

@retry(wait=wait, stop=stop)
def rewrite_query(question: str, history: list = []) -> str:
    """
    Elite Search Architect: 
    Focuses on Multi-Perspective Expansion and Entity Anchoring.
    """
    history_context = history[-4:] if history else "No previous history."

    # Force the LLM to follow the new Elite Rules
    prompt = f"""
    You are a Lead Search Architect for the Hoopball League. 
    Expand the user's question into a STANDALONE multi-perspective search query.

    NEW ELITE RULES:
    1. ENTITY ANCHORING: If a player is mentioned, include their "Full Name", "Last Name", and "Team Name".
    2. SYNONYM INJECTION: Add related statistical terms (e.g., for 'best', add 'scoring leader PPG stats rank').
    3. RANK RESOLUTION: Convert ranks (e.g., '67th') into absolute search terms combined with the category.
    4. NOISE REMOVAL: Strip all conversational filler but keep ALL technical keywords.
    5. OUTPUT ONLY: Return keywords separated by spaces. 

    Conversation History:
    {history_context}

    New User Question: {question}

    Standalone Search Keywords:"""

    response = completion(model=MODEL, messages=[{"role": "user", "content": prompt}], **model_kwargs())
    rewritten = response.choices[0].message.content.strip().replace('"', '')

    # DEBUG: You will see much richer queries now
    if DEBUG:
        print(f"\n[DEBUG] --- Elite Query Optimization ---")
        print(f"      Original: '{question}'")
        print(f"      Optimized: '{rewritten}'")
    
    return rewritten


def make_rag_messages(question, clean_history, chunks):
    """Constructs the message array for the final LLM completion."""
    context_text = "\n\n".join(
        f"--- Source: {chunk.metadata.get('source', 'League Archive')} ---\n{chunk.page_content}" 
        for chunk in chunks
    )
    
    # Constructing a clean chat sequence: System (with Context) -> History -> Current Question
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context_text)}]
    messages.extend(clean_history)
    messages.append({"role": "user", "content": question})
    return messages

def fetch_context(original_question, clean_history=[], use_rewrite=True, use_rerank=True):
    """Advanced retrieval: Query Rewriting -> Vector Search -> Merging -> Reranking.

    use_rewrite / use_rerank let the evaluation measure each stage's contribution (ablation).
    """
    chunks1 = fetch_context_unranked(original_question)
    chunks2 = []
    if use_rewrite:
        # Use history-aware rewriting for better context memory
        rewritten_question = rewrite_query(original_question, clean_history)
        chunks2 = fetch_context_unranked(rewritten_question)
    
    # Merge and deduplicate chunks based on content
    merged = chunks1[:]
    seen_content = {c.page_content for c in chunks1}
    for chunk in chunks2:
        if chunk.page_content not in seen_content:
            merged.append(chunk)
            seen_content.add(chunk.page_content)
            
    if not use_rerank:
        # Interleave both result lists; merged[:FINAL_K] would contain only original-query
        # results, so the rewrite would have no effect without the reranker.
        interleaved, seen = [], set()
        for pair in zip_longest(chunks1, chunks2):
            for chunk in pair:
                if chunk is not None and chunk.page_content not in seen:
                    interleaved.append(chunk)
                    seen.add(chunk.page_content)
        return interleaved[:FINAL_K]
    # Apply reranking to find the top FINAL_K chunks
    if RERANKER == "local":
        return rerank_local(original_question, merged)[:FINAL_K]
    return rerank(original_question, merged)[:FINAL_K]


def fetch_context_unranked(question):
    """Standard vector search in ChromaDB."""
    try:
        query_embedding = openai.embeddings.create(model=embedding_model, input=[question]).data[0].embedding
        results = collection.query(query_embeddings=[query_embedding], n_results=RETRIEVAL_K)
        
        chunks = []
        if results["documents"] and results["documents"][0]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                chunks.append(Result(page_content=doc, metadata=meta))
        return chunks
    except Exception as e:
        print(f"[DEBUG] Vector search error: {e}")
        return []

@retry(wait=wait, stop=stop)
def answer_question(question: str, history: list[dict] = [], use_rewrite=True, use_rerank=True) -> tuple[str, list]:
    """
    Main entry point for the RAG logic. 
    Processes history, retrieves context, and generates a professional answer.
    """
    # 1. Clean Gradio's complex history structure
    context_window = history[-4:]
    clean_history = clean_gradio_history(context_window)
    
    # 2. Retrieve the best context using the rewritten query
    chunks = fetch_context(question, clean_history, use_rewrite=use_rewrite, use_rerank=use_rerank)
    
    # 3. Generate final prompt and send to LLM
    messages = make_rag_messages(question, clean_history, chunks)
    
    response = completion(model=MODEL, messages=messages, temperature=0, **model_kwargs())
    return response.choices[0].message.content, chunks
