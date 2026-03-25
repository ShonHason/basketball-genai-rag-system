import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# הגדרות נתיבים
VECTOR_DB_PATH = "vector_db/unified_index"
DATA_PATH = "data"

def setup_env():
    load_dotenv(override=True)
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set")

def load_and_enhance_documents():
    """טוען מסמכים ומזריק הקשר למטא-דאטה"""
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    
    raw_documents = loader.load()
    enhanced_docs = []

    for doc in raw_documents:
        path = Path(doc.metadata["source"])
        # זיהוי סוג המסמך לפי התיקייה (players/teams/league)
        doc_type = path.parts[1] 
        
        # הפיכת שם הקובץ (למשל caleb_brooks) לשם נקי (Caleb Brooks)
        entity_name = path.stem.replace("_", " ").title()
        
        doc.metadata["entity_name"] = entity_name
        doc.metadata["doc_type"] = doc_type
        
        enhanced_docs.append(doc)
    
    return enhanced_docs

def split_with_context(documents):
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    
    # splitter ייעודי לטבלאות - חותך לפי מספר תווים שמתאים לכ-25 שורות
    stats_text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    final_chunks = []

    for doc in documents:
        path = str(doc.metadata.get("source", ""))
        is_stats_file = "league_stats" in path
        
        header_chunks = markdown_splitter.split_text(doc.page_content)
        
        for chunk in header_chunks:
            chunk.metadata.update(doc.metadata)
            entity = chunk.metadata.get('entity_name', 'Unknown')
            dtype = chunk.metadata.get('doc_type', 'General')
            section = chunk.metadata.get('Header 2', 'General Info')
            
            context_prefix = f"Subject: {entity} | Category: {dtype} | Section: {section}"
            
            if is_stats_file:
                # חותך טבלאות ארוכות ל-2 או 3 חלקים, אבל מזריק לכל חלק את ה-Prefix
                # ככה "מקום 45" יהיה בצ'אנק שמתחיל במילים "Subject: Top Scorers"
                sub_chunks = stats_text_splitter.split_documents([chunk])
                for sub in sub_chunks:
                    sub.page_content = f"{context_prefix}\nContent: {sub.page_content}"
                    final_chunks.append(sub)
            else:
                # טיפול רגיל במסמכי שחקנים
                chunk.page_content = f"{context_prefix}\nContent: {chunk.page_content}"
                sub_chunks = text_splitter.split_documents([chunk])
                final_chunks.extend(sub_chunks)

    return final_chunks

def create_unified_vector_db(chunks):
    """שמירה ל-Vector DB אחד חזק"""
    if os.path.exists(VECTOR_DB_PATH):
        shutil.rmtree(VECTOR_DB_PATH)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    print(f"✅ Success! Unified DB created with {vectorstore._collection.count()} chunks.")

def build_knowledge_base():
    setup_env()
    print("🚀 Loading documents...")
    documents = load_and_enhance_documents()
    print("✂️ Splitting documents with context...")
    chunks = split_with_context(documents)
    print("💾 Creating Vector DB...")
    create_unified_vector_db(chunks)

if __name__ == "__main__":
    build_knowledge_base()