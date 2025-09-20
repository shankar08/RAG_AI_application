from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain_community.llms import OpenAI
from langchain.schema import AIMessage, HumanMessage
import gradio as gr
import traceback

from Agentic_ai_with_RAG import vectorStore  # Import the pre-initialized vectorStore

# Toggle this flag to True to enable debug logs
DEBUG = True

def predict(message, history):
    try:
        if DEBUG: print(f"\n📥 Received message: {message}")
        history_langchain_format = []

        # Convert chat history into LangChain format
        for human, ai in history:
            history_langchain_format.append(HumanMessage(content=human))
            history_langchain_format.append(AIMessage(content=ai))

        llm = OpenAI()
        # Define Prompt
        template = """You are a medical assistant chatbot helping answer patient questions based only on the provided context.
        Do not guess or provide inaccurate information. If the answer is not found in the context, say you don’t know.
        You will answer the question based on the context - {context}.
        Question: {question}
        Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        # Create Retrieval QA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorStore.as_retriever(search_type="similarity", k=4),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

        # Run the chain with the user query
        result = qa_chain({"query": message})
        answer = result['result']

        if DEBUG: print(f"✅ Retrieved Answer:\n{answer}")

        # Update history
        history_langchain_format.append(HumanMessage(content=message))
        history_langchain_format.append(AIMessage(content=answer))

        return answer

    except Exception as e:
        error_trace = traceback.format_exc()
        if DEBUG: print(f"❌ Exception occurred:\n{error_trace}")
        return "⚠️ Sorry, an internal error occurred. Please try again later."

# Gradio Chatbot UI
gr.ChatInterface(
    fn=predict,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Ask me a question related to Healthcare and Medical Services", container=False, scale=7),
    title="Neil the Medical Assistant",
    theme="soft",
    examples=[
        "What are the main differences between acute and chronic medical conditions?",
        "What are the main differences between acute and chronic medical conditions?",
        "What does mild concentric LV hypertrophy mean?"
    ],
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
).launch(share=True)