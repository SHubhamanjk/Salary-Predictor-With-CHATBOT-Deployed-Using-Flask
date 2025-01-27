import numpy as np
import pandas as pd
import pickle
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def upload_resume(file):
    """
    Upload and extract text from a resume file (PDF).
    Args:
        file: The uploaded file object.
    Returns:
        Extracted text or an error message.
    """
    if file.filename == '':
        return {"message": "No file selected!"}, 400

    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        if not text.strip():
            return {"message": "The PDF is empty or unreadable!"}, 400
        return text
    except Exception as e:
        return {"message": f"Error reading the PDF: {e}"}, 500

def process_resume(text):
    """
    Process the resume text into retrievable chunks using embeddings.
    Args:
        text: The extracted text from the resume.
    Returns:
        A retriever object for conversational chains.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(chunks, embeddings)
        retriever = vector_store.as_retriever()
        return retriever
    except Exception as e:
        raise RuntimeError(f"Error processing resume text: {e}")

def load_groq():
    """
    Load the ChatGroq model with the API key from environment variables.
    Returns:
        A ChatGroq instance.
    """
    try:
        groq = ChatGroq(api_key=os.environ.get('GROQ_API_KEY'))
        return groq
    except Exception as e:
        raise RuntimeError(f"Error loading ChatGroq: {e}")

def generate_response(groq, query):
    """
    Generate a response from the ChatGroq model for a given query.
    Args:
        groq: The ChatGroq instance.
        query: The input query string.
    Returns:
        The response string from ChatGroq.
    """
    try:
        response = groq.query(query)
        return response
    except Exception as e:
        raise RuntimeError(f"Error generating response: {e}")

def create_conversational_chain(retriever, groq):
    """
    Create a conversational retrieval chain.
    Args:
        retriever: A retriever object for fetching relevant text chunks.
        groq: The ChatGroq instance.
    Returns:
        A conversational retrieval chain.
    """
    try:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        chain = ConversationalRetrievalChain.from_llm(llm=groq, retriever=retriever, memory=memory)
        return chain
    except Exception as e:
        raise RuntimeError(f"Error creating conversational chain: {e}")
