import os
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_chroma import Chroma

# הגדרות
VECTOR_DB_PATH = "vector_db/unified_index"
MODEL = "gpt-4o" 

load_dotenv(override=True)

def initialize_rag():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    unified_db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
    llm = ChatOpenAI(model=MODEL, temperature=0)
    return unified_db, llm

unified_db, llm = initialize_rag()

# -----------------------------
# ה-Rewriter המשופר - המפתח לפתרון
# -----------------------------
def rewrite_question(question, history):
    history_context = history[-3:] if history else "No previous history."

    prompt = f"""
You are an expert Query Optimizer for a Basketball RAG system. 
Your goal is to rewrite the user's question into a standalone search query that retrieves ALL necessary documents.

RULES FOR RANKINGS & DETAILS:
1. If the user mentions a RANK (e.g., "71st", "6th", "10th") and asks for PERSONAL details (e.g., "starter", "hobbies", "biography"):
   - The query MUST include the rank (e.g., "6th in scoring").
   - The query MUST ALSO include the terms: "player personal profile", "starter status", and "full biography".
   - This ensures the Vector DB retrieves BOTH the ranking table AND the individual player files.

RULES FOR CONTEXT:
- Replace pronouns (he, him, his, "the same question") with the actual names, ranks, or categories from the history.
- If the question is "and 67th?", rewrite it as: "Who is the player ranked 67th in [Category] and what is his nationality and starter status?"

Conversation History:
{history_context}

New User Question: {question}

Final Standalone Search Query:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()
    

def get_system_prompt(context):
    return f"""
You are an expert assistant for the Hoopball National League. 
Answer the user's question using the provided context.

RULES:
- If the context contains a "Championship History" or "Championship Table", use it for questions about winners.
- If you don't know the answer, say "I don't have that information in the database."
- Keep answers factual based on the context.

CONTEXT:
{context}
"""

def answer_question(question: str, history):
    # 1. Generate standalone query
    standalone_query = rewrite_question(question, history)
    print(f"\n🔍 Standalone Query: {standalone_query}")

    # 2. Retrieval
    docs_with_scores = unified_db.similarity_search_with_score(standalone_query, k=8)

    if not docs_with_scores:
        return "I don't have that information in the database."

    # --- CRITICAL FIX: Define 'context' here ---
    context_parts = []
    for doc, score in docs_with_scores:
        context_parts.append(doc.page_content)
    
    context = "\n\n".join(context_parts) # This was likely missing or out of scope
    # ------------------------------------------

    # 3. Create messages for LLM
    messages = [SystemMessage(content=get_system_prompt(context))]
    
    # Fix for 'too many values to unpack' (Dictionary History)
    for msg in history[-10:]:
        if isinstance(msg, dict):
            if msg['role'] == 'user':
                messages.append(HumanMessage(content=msg['content']))
            else:
                messages.append(AIMessage(content=msg['content']))
        else:
            # Fallback for older Gradio versions (Tuples)
            messages.append(HumanMessage(content=msg[0]))
            messages.append(AIMessage(content=msg[1]))

    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)
    return response.content


def gradio_wrapper(message, history):
    # In Gradio < 5.0, history is already [(user, bot), (user, bot)]
    return answer_question(message, history)

if __name__ == "__main__":
    demo = gr.ChatInterface(
        fn=gradio_wrapper,
        title="NLA Smart Assistant"
        # Removed type="messages"
    )
    demo.launch(inbrowser=True)