import tempfile
import requests
import fitz  # PyMuPDF
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
import os
from dotenv import load_dotenv

load_dotenv()

# Load from .env
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def download_pdf_from_url(blob_url: str) -> str:
    response = requests.get(blob_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download PDF. Status code: {response.status_code}")
    
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_file.write(response.content)
    tmp_file.close()
    return tmp_file.name


def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n", " ", ""]
    )
    return splitter.split_text(text)


def build_faiss_index(chunks: List[str]) -> FAISS:
    documents = [Document(page_content=chunk) for chunk in chunks]
    embedding = OpenAIEmbeddings(
        openai_api_base=OPENAI_API_BASE,
        openai_api_key=OPENAI_API_KEY
    )
    vectorstore = FAISS.from_documents(documents, embedding)
    return vectorstore


def query_faiss(vectorstore: FAISS, question: str) -> str:
    retriever = vectorstore.as_retriever()
    relevant_docs = retriever.get_relevant_documents(question)

    llm = ChatOpenAI(
        temperature=0,
        model=OPENAI_MODEL,
        openai_api_base=OPENAI_API_BASE,
        openai_api_key=OPENAI_API_KEY
    )

    chain = load_qa_chain(llm, chain_type="stuff")
    result = chain.run(input_documents=relevant_docs, question=question)
    return result


def process_blob_and_answer(blob_url: str, question: str) -> str:
    if not OPENAI_API_KEY:
        raise EnvironmentError("Missing OPENAI_API_KEY in environment variables.")
    
    pdf_path = download_pdf_from_url(blob_url)
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    vectorstore = build_faiss_index(chunks)
    answer = query_faiss(vectorstore, question)
    return answer
