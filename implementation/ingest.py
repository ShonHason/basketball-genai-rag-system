import os
import glob
import shutil
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings


from dotenv import load_dotenv

MODEL = "gpt-4.1-mini"

DB_NAME = str(Path(__file__).parent.parent / "vector_db")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "data")

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

load_dotenv(override=True)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def create_embeddings(chunks):
    print(f"📦 Processing {len(chunks):,} total chunks...") # NEW PRINT

    if os.path.exists(DB_NAME):
        # Improved deletion logic
        shutil.rmtree(DB_NAME)
        print("🧹 Old vector DB directory removed for a clean start")

    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_NAME
    )

    # Access the collection to verify the math
    collection = vectorstore._collection
    count = collection.count()

    # Retrieve a sample to check dimensions
    sample = collection.get(limit=1, include=["embeddings"])
    if sample["embeddings"]:
        sample_embedding = sample["embeddings"][0]
        dimensions = len(sample_embedding)
        print("-" * 50)
        print(f"📊 VECTOR STORE SUMMARY")
        print(f"🔹 Total Chunks (Vectors): {count:,}")
        print(f"🔹 Embedding Model: {embeddings.model}")
        print(f"🔹 Vector Dimensions: {dimensions:,}")
        print("-" * 50)
    else:
        print("⚠️ Database created but no embeddings found.")

    return vectorstore

def fetch_documents():
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents


def create_chunks(documents):
    # 1. Define the headers to split on (matches the # and ## in your stats files)
    headers_to_split_on = [ 
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # 2. Final text splitter for the actual content
    # Increased chunk_size to 1000 to keep tables more intact
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, 
        chunk_overlap=100,
        separators=["\n\n", "\n", "|", " ", ""] # הוספת "|" עוזרת לא לחתוך טבלאות באמצע שורה
    )

    all_chunks = []
    
    for doc in documents:
        # First split by markdown headers to keep context
        header_splits = markdown_splitter.split_text(doc.page_content)
        
        # Then split those into manageable chunks while keeping metadata
        for split in header_splits:
            # Transfer original document metadata (doc_type, etc.) to the new splits
            split.metadata.update(doc.metadata)
            
        final_splits = text_splitter.split_documents(header_splits)
        all_chunks.extend(final_splits)
        
    return all_chunks

def create_embeddings(chunks):
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_NAME
    )

    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")
