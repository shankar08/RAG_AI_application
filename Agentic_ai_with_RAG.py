import warnings
import os
import tiktoken
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Setup open ai model
from langchain_community.llms import HuggingFaceHub

from langchain_community.llms import OpenAI

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
import gradio as gr
import traceback

warnings.filterwarnings("ignore")
load_dotenv()

key = os.environ.get("OPENAI_API_KEY")
# print(key)


# Read and clean the dataset
with open('ai-medical-chatbot.txt', 'r') as f:
    data = f.read()

data = data.replace('\n\n','\n') # Clean unnecessary line breaks
data = data.split('---') # Split the data into separate sections based on a delimiter

# print(data[4])


for i in range(0, len(data)):
    if i==4:
      data[i] = data[i].replace('\n**', '\n###').replace('**','')
    else:
      data[i] = data[i].replace('**','')

# print(data[4])

# Organize the data into question-answer pairs
ques_ans = dict()
for i in range(0, len(data)):
    topics = data[i].split('\n###')
    for topic in topics[1:]:
      question_answer_pair = topic.split('\n')
      ques_ans[question_answer_pair[0]] = " ".join(question_answer_pair[1:])

# print(len(ques_ans))

# print(ques_ans.keys())

# print(ques_ans[" What are Medical Conditions?"])

all_content = str()
for key, value in ques_ans.items():
    # print(key)
    all_content += key + " " + value + "\n"

# print(all_content)

# Text Chunking
# Split the large content into smaller chunks for indexing
text_splitter = CharacterTextSplitter(separator='\n', chunk_size=300, chunk_overlap=128, length_function=len)

chunks = text_splitter.split_text(all_content)

# Display the first chunk for verification
# print(chunks)


# # Store embedding in FAISS vector store
embeddings = OpenAIEmbeddings()

vectorStore = FAISS.from_texts(chunks, embeddings)
vectorStore.save_local("faiss_doc_idx")

#  Perform Semantic search
# docs = vectorStore.similarity_search("What are the main differences between acute and chronic medical conditions?")
# for doc in docs:
#     print(doc.page_content)

# Toggle this flag to True to enable debug logs
DEBUG = True

# Setup open ai model
def predict(message, history):
    try:
        if DEBUG: print(f"\n📥 Received message: {message}")
        history_langchain_format = []

        # Convert chat history into LangChain format
        for human, ai in history:
            history_langchain_format.append(HumanMessage(content=human))
            history_langchain_format.append(AIMessage(content=ai))

        llm = OpenAI
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
    title="DocumentQABot",
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