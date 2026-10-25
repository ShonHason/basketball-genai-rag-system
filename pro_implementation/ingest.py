import os
import argparse
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from tqdm import tqdm
from litellm import completion
from multiprocessing import Pool
from tenacity import retry, wait_exponential
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)

MODEL = "openai/gpt-4.1-mini" 

BASE_DIR = Path(__file__).parent.parent 
DB_NAME = str(BASE_DIR / "preprocessed_db")
collection_name = "docs"
embedding_model = "text-embedding-3-large"

KNOWLEDGE_BASE_PATH = BASE_DIR / "data" 

AVERAGE_CHUNK_SIZE = 800
wait = wait_exponential(multiplier=1, min=10, max=240)
WORKERS = 3
openai = OpenAI()

class Result(BaseModel):
    page_content: str
    metadata: dict

class Chunk(BaseModel):
    headline: str = Field(description="Heading for the chunk")
    summary: str = Field(description="Summary of the chunk")
    original_text: str = Field(description="The original text")

    def as_result(self, document):
        return Result(
            page_content=f"{self.headline}\n\n{self.summary}\n\n{self.original_text}",
            metadata={"source": document["source"], "type": document["type"]},
        )

class Chunks(BaseModel):
    chunks: list[Chunk]

def fetch_documents():
    if not KNOWLEDGE_BASE_PATH.exists():
        print(f"❌ ERROR: Directory NOT FOUND at {KNOWLEDGE_BASE_PATH}")
        return []

    documents = []
    # סורק את כל קבצי ה-md בתוך תיקיית data
    for file in KNOWLEDGE_BASE_PATH.rglob("*.md"):
        # אם הקובץ בתוך תת-תיקייה, נשתמש בשם שלה כ-type, אחרת 'general'
        doc_type = file.parent.name if file.parent != KNOWLEDGE_BASE_PATH else "league_data"
        with open(file, "r", encoding="utf-8") as f:
            documents.append({
                "type": doc_type, 
                # Relative path, not file.name: teams/x.md and team_summaries/x.md share a filename
                "source": str(file.relative_to(KNOWLEDGE_BASE_PATH)), 
                "text": f.read()
            })

    print(f"✅ Loaded {len(documents)} documents from 'data' folder")
    return documents

def make_prompt(document):
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    # עדכון ההקשר ל-Hoopball League
    return f"""
You are an expert data architect for the Hoopball League.
Split this document into at least {how_many} overlapping chunks for a basketball knowledge base.

IMPORTANT: Ensure each chunk includes a 100-word overlap from the end of the previous chunk to maintain context.

Document Type: {document["type"]}
Source: {document["source"]}

Ensure each chunk captures specific details like player names, game stats, or team info.
Respond with headlines, summaries, and original text.

Document:
{document["text"]}
"""

# ... שאר הפונקציות (process_document, create_chunks, create_embeddings) נשארות אותו דבר ...

@retry(wait=wait)
def process_document(document):
    messages = [{"role": "user", "content": make_prompt(document)}]
    response = completion(model=MODEL, messages=messages, response_format=Chunks)
    reply = response.choices[0].message.content
    doc_as_chunks = Chunks.model_validate_json(reply).chunks
    return [chunk.as_result(document) for chunk in doc_as_chunks]

def create_chunks(documents):
    chunks = []
    with Pool(processes=WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(process_document, documents), total=len(documents)):
            chunks.extend(result)
    return chunks

def create_fixed_chunks(documents, chunk_size):
    """Deterministic chunking (no LLM) - used to compare chunk sizes."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size // 8,
        separators=["\n## ", "\n\n", "\n", " ", ""],
    )
    chunks = []
    for document in documents:
        for text in splitter.split_text(document["text"]):
            chunks.append(Result(page_content=text, metadata={"source": document["source"], "type": document["type"]}))
    return chunks

def create_embeddings(chunks, name=collection_name):
    chroma = PersistentClient(path=DB_NAME)
    if name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(name)

    collection = chroma.get_or_create_collection(name)
    
    texts = [chunk.page_content for chunk in chunks]
    metas = [chunk.metadata for chunk in chunks]
    all_vectors = []
    
    # התיקון: שליחה ב"מנות" של 100 כדי לא לעבור את מגבלת ה-API
    batch_size = 100 
    print(f"Creating embeddings for {len(chunks)} chunks...")
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i + batch_size]
        # כאן ה-API מקבל רק 100 שורות בכל פעם ולא אלפים
        emb_res = openai.embeddings.create(model=embedding_model, input=batch_texts).data
        vectors = [e.embedding for e in emb_res]
        all_vectors.extend(vectors)

    # הוספה ל-ChromaDB
    ids = [str(i) for i in range(len(chunks))]
    collection.add(ids=ids, embeddings=all_vectors, documents=texts, metadatas=metas)
    print(f"✅ Vectorstore '{name}' created successfully with {collection.count()} chunks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, help="Fixed-size chunking (no LLM) into collection docs_cs<N>")
    parser.add_argument("--collection", help="Collection name (default: docs, or docs_cs<N> with --chunk-size)")
    args = parser.parse_args()

    docs = fetch_documents()
    if docs:
        if args.chunk_size:
            all_chunks = create_fixed_chunks(docs, args.chunk_size)
            name = args.collection or f"docs_cs{args.chunk_size}"
        else:
            all_chunks = create_chunks(docs)
            name = args.collection or collection_name
        create_embeddings(all_chunks, name)
        print("🚀 Ingestion complete!")
    else:
        print("⚠️ Check if your .md files are actually inside the 'data' folder.")