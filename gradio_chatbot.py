import os
import traceback
import warnings
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import gradio as gr

from Agentic_ai_with_RAG import build_vector_store   # ✅ import the function

warnings.filterwarnings("ignore")
load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")

# ── Rebuild FAISS index every time chatbot starts ──────────────────────────────
print("🔄 Rebuilding vector store...")
vectorStore = build_vector_store()                   # ✅ always fresh index

# ── Build LCEL RAG chain ───────────────────────────────────────────────────────
retriever = vectorStore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=api_key,
    temperature=0.2
)

prompt = PromptTemplate.from_template("""
You are a medical assistant chatbot helping answer patient questions \
based only on the provided context.
Do not guess or provide inaccurate information.
If the answer is not found in the context, say "I'm sorry, I don't have \
enough information to answer that question."

Context: {context}

Question: {question}

Answer:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── Gradio predict function ────────────────────────────────────────────────────
DEBUG = True

def predict(message, history):
    try:
        if DEBUG:
            print(f"\n📥 Received message: {message}")
        answer = qa_chain.invoke(message)
        if DEBUG:
            print(f"✅ Answer: {answer}")
        return answer
    except Exception as e:
        traceback.print_exc()
        return "⚠️ Sorry, an internal error occurred. Please try again later."

# ── Gradio UI ──────────────────────────────────────────────────────────────────
demo = gr.ChatInterface(
    fn=predict,
    type="messages",
    chatbot=gr.Chatbot(
        height=400,
        type="messages",
        placeholder="<strong> Medical Assistant</strong><br>Ask me any medical question!"
    ),
    textbox=gr.Textbox(
        placeholder="Ask me a question related to Healthcare and Medical Services...",
        container=False,
        scale=7
    ),
    title="🩺 Medical Assistant",
    description="An AI-powered medical assistant. Ask questions about symptoms, conditions, and treatments.",
    theme="soft",
    examples=[
        "What are the main differences between acute and chronic medical conditions?",
        "What does mild concentric LV hypertrophy mean?",
        "What are common symptoms of diabetes?",
        "What is the difference between Type 1 and Type 2 diabetes?",
        "What are the warning signs of a heart attack?"
    ],
)

if __name__ == "__main__":
    demo.launch(share=True)
