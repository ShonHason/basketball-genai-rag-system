import os
from openai import OpenAI
from dotenv import load_dotenv
from chromadb import PersistentClient
from litellm import completion
from pydantic import BaseModel, Field
from pathlib import Path
from tenacity import retry, wait_exponential

load_dotenv(override=True)

# --- Configuration ---
MODEL = "openai/gpt-4o-mini"
import litellm
litellm.suppress_debug_info = True
litellm.set_verbose = False
#MODEL = "openrouter/openai/gpt-oss-120b" 
DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
collection_name = "docs"
embedding_model = "text-embedding-3-large"
#wait = wait_exponential(multiplier=1, min=10, max=240)
wait = wait_exponential(multiplier=2, min=10, max=60)
# Initialize Chroma and OpenAI clients
openai = OpenAI()
chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(collection_name)

RETRIEVAL_K = 15
FINAL_K = 7

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

@retry(wait=wait)
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
            temperature=0 
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

@retry(wait=wait)
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

    response = completion(model=MODEL, messages=[{"role": "user", "content": prompt}])
    rewritten = response.choices[0].message.content.strip().replace('"', '')

    # DEBUG: You will see much richer queries now
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

def fetch_context(original_question, clean_history=[]):
    """Advanced retrieval: Query Rewriting -> Vector Search -> Merging -> Reranking."""
    # Use history-aware rewriting for better context memory
    rewritten_question = rewrite_query(original_question, clean_history)
    
    # Perform dual search (original and rewritten)
    chunks1 = fetch_context_unranked(original_question)
    chunks2 = fetch_context_unranked(rewritten_question)
    
    # Merge and deduplicate chunks based on content
    merged = chunks1[:]
    seen_content = {c.page_content for c in chunks1}
    for chunk in chunks2:
        if chunk.page_content not in seen_content:
            merged.append(chunk)
            seen_content.add(chunk.page_content)
            
    # Apply reranking to find the top FINAL_K chunks
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

@retry(wait=wait)
def answer_question(question: str, history: list[dict] = []) -> tuple[str, list]:
    """
    Main entry point for the RAG logic. 
    Processes history, retrieves context, and generates a professional answer.
    """
    # 1. Clean Gradio's complex history structure
    context_window = history[-4:]
    clean_history = clean_gradio_history(context_window)
    
    # 2. Retrieve the best context using the rewritten query
    chunks = fetch_context(question, clean_history)
    
    # 3. Generate final prompt and send to LLM
    messages = make_rag_messages(question, clean_history, chunks)
    
    response = completion(model=MODEL, messages=messages)
    return response.choices[0].message.content, chunks