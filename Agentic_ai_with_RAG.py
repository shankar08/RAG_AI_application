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