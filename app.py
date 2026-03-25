import gradio as gr
from dotenv import load_dotenv
import sys # להדפסה מיידית לטרמינל
from pro_implementation.answer import answer_question

load_dotenv(override=True)

def format_context(context):
    if not context:
        return "*No context retrieved.*"
    
    result = "## 🏀 Relevant League Context\n\n"
    for doc in context:
        source = doc.metadata.get('source', 'League Archive')
        result += f"**Source:** {source}\n\n"
        result += doc.page_content + "\n\n---\n\n"
    return result

def chat(history):
    """ניהול שיחה עם חילוץ טקסט תקין ל-Gradio 6.0"""
    # חילוץ ההודעה האחרונה
    raw_message = history[-1]["content"]
    
    # בדיקה אם ההודעה היא בפורמט החדש (רשימה) וחילוץ הטקסט
    if isinstance(raw_message, list):
        # מחלץ את ה-text מתוך [{'text': '...', 'type': 'text'}]
        last_message = next((item["text"] for item in raw_message if item["type"] == "text"), "")
    else:
        last_message = raw_message

    print(f"\n[LOG] User asked: '{last_message}'")
    print(f"[LOG] Sending history of {len(history)-1} messages to RAG...")

    try:
        # שליחת הטקסט הנקי ל-RAG
        answer, context = answer_question(last_message, history[:-1])
        print(f"[LOG] RAG Success: Found {len(context)} chunks.")
        
    except Exception as e:
        print(f"[ERROR] RAG failed with exception: {str(e)}")
        error_msg = f"Sorry, an error occurred: {str(e)}"
        history.append({"role": "assistant", "content": error_msg})
        return history, "## ❌ Error during retrieval"

    history.append({"role": "assistant", "content": answer})
    return history, format_context(context)

def main():
    def put_message_in_chatbot(message, history):
        if history is None:
            history = []
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Inter", "sans-serif"])

    with gr.Blocks() as ui:
        gr.Markdown("# 🏀 Hoopball League Expert Assistant\nAsk me anything about players, teams, or stats!")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="💬 Conversation", 
                    height=600
                )
                message = gr.Textbox(
                    placeholder="Ask about Elias Vega or the Madrid Quasar...",
                    show_label=False,
                )

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
                    label="📚 Scanned Documents"
                )

        message.submit(
            put_message_in_chatbot, 
            inputs=[message, chatbot], 
            outputs=[message, chatbot]
        ).then(
            chat, 
            inputs=chatbot, 
            outputs=[chatbot, context_markdown]
        )

    ui.launch(inbrowser=True, theme=theme)

if __name__ == "__main__":
    main()