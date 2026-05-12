import warnings
import os
import tiktoken
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

def build_vector_store():
    """Reads the dataset, chunks it, creates embeddings and saves FAISS index."""
    api_key = os.environ.get("OPENAI_API_KEY")

    with open('ai-medical-chatbot.txt', 'r') as f:
        data = f.read()

    data = data.replace('\n\n', '\n')
    data = data.split('---')

    for i in range(0, len(data)):
        if i == 4:
            data[i] = data[i].replace('\n**', '\n###').replace('**', '')
        else:
            data[i] = data[i].replace('**', '')

    ques_ans = dict()
    for i in range(0, len(data)):
        topics = data[i].split('\n###')
        for topic in topics[1:]:
            question_answer_pair = topic.split('\n')
            ques_ans[question_answer_pair[0]] = " ".join(question_answer_pair[1:])

    all_content = str()
    for key, value in ques_ans.items():
        all_content += key + " " + value + "\n"

    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=300,
        chunk_overlap=128,
        length_function=len
    )
    chunks = text_splitter.split_text(all_content)

    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectorStore = FAISS.from_texts(chunks, embeddings)
    vectorStore.save_local("faiss_doc_idx")

    print("✅ Vector store rebuilt and saved!")
    return vectorStore


# Allow running standalone too
if __name__ == "__main__":
    build_vector_store()
