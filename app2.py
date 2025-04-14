import streamlit as st
import os
import time
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import openai
from dotenv import load_dotenv

# Fix for the HuggingFaceEmbeddings import
from langchain_huggingface import HuggingFaceEmbeddings

# Page configuration
st.set_page_config(
    page_title="Research Papers RAG",
    page_icon="📚",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Custom CSS for better styling with dark theme compatibility
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #7FDBFF;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #1E1E1E;
        color: #FFFFFF;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #333333;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
    .response-area {
        background-color: #2D2D2D;
        color: #FFFFFF;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .source-document {
        background-color: #232323;
        color: #FFFFFF;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)


def check_api_keys():
    missing_keys = []
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        missing_keys.append("GROQ_API_KEY")
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        missing_keys.append("HF_TOKEN")
    return missing_keys, groq_api_key, hf_token


def create_vector_embedding(uploaded_files):
    with st.spinner("Processing documents... This might take a while ⏳"):
        try:
            st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            docs = []
            for uploaded_file in uploaded_files:
                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.read())
                loader = PyPDFLoader(uploaded_file.name)
                docs.extend(loader.load())

            if not docs:
                return False, "No documents were loaded. Please check your PDF files."

            st.session_state.total_docs = len(docs)
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(docs)
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents,
                st.session_state.embeddings
            )
            return True, f"Successfully processed {len(docs)} documents into {len(st.session_state.final_documents)} chunks"

        except Exception as e:
            return False, f"Error creating embeddings: {str(e)}"


def process_query(query):
    try:
        missing_keys, groq_api_key, _ = check_api_keys()
        if "GROQ_API_KEY" in missing_keys:
            return False, "Missing GROQ_API_KEY in environment variables"

        llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the question based on the provided context only.
            If the answer is not in the context, say "I don't have enough information to answer this question."
            Provide a clear, concise response that directly addresses the question.

            <context>
            {context}
            </context>

            Question: {input}
            """
        )
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 5})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': query})
        processing_time = time.process_time() - start

        return True, {
            "answer": response['answer'],
            "context": response['context'],
            "processing_time": processing_time
        }

    except Exception as e:
        return False, f"Error processing query: {str(e)}"


def main():
    st.markdown('<h1 class="main-header">📚 Research Papers Q&A Assistant</h1>', unsafe_allow_html=True)
    missing_keys, groq_api_key, hf_token = check_api_keys()
    if missing_keys:
        st.error(f"Missing required API keys: {', '.join(missing_keys)}")
        st.info("Please set the required environment variables in your .env file.")
        return

    with st.sidebar:
        st.markdown('<h2 class="sub-header">About</h2>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        This application uses Retrieval-Augmented Generation (RAG) to answer questions 
        about your research papers. It processes PDF documents, creates vector embeddings, 
        and uses Llama3 via Groq to generate accurate responses based on your documents.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<h2 class="sub-header">Instructions</h2>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        1. Upload one or more PDF files.
        2. Click "Process Documents" to create embeddings from your PDFs.
        3. Enter your question about the research papers.
        4. Click "Submit Question" to get an answer.
        </div>
        """, unsafe_allow_html=True)

        if "vectors" in st.session_state:
            st.success(f"✅ Documents processed")
            st.success(f"✅ Document chunks created: {len(st.session_state.final_documents)}")

    st.markdown('<h2 class="sub-header">Document Upload</h2>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files and st.button("Process Documents", key="process_docs"):
        success, message = create_vector_embedding(uploaded_files)
        if success:
            st.success(message)
        else:
            st.error(message)

    st.markdown('<h2 class="sub-header">Ask Questions</h2>', unsafe_allow_html=True)
    user_prompt = st.text_area("Enter your question about the research papers:", height=100, 
                              placeholder="e.g., What are the main findings related to climate change in these papers?")

    if st.button("Submit Question", key="submit_query"):
        if "vectors" not in st.session_state:
            st.warning("Please process the documents first by uploading and clicking 'Process Documents'.")
        elif not user_prompt:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer... This might take a few seconds ⏳"):
                success, result = process_query(user_prompt)
                if success:
                    st.markdown(f"<div class='response-area'><h3>Answer:</h3>{result['answer']}</div>", unsafe_allow_html=True)
                    st.info(f"Processing time: {result['processing_time']:.2f} seconds")
                    with st.expander("View Source Documents"):
                        for i, doc in enumerate(result['context']):
                            st.markdown(f"<div class='source-document'><strong>Source {i+1}:</strong><br>{doc.page_content}</div>", unsafe_allow_html=True)
                else:
                    st.error(result)


if __name__ == "__main__":
    main()