import streamlit as st
#import os
#from dotenv import load_dotenv
#LangChain Imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pathlib import Path


# =========================
# Page Config
# =========================
st.set_page_config(page_title="College AI Assistance Chatbot")
st.title("College AI Assistance Chatbot")
st.write("Ask any question related to College AI Assistance")

# Cache Document Loading
# ----------------------

DATA_DIR = Path(
    r"C:\Users\samru\Documents\Codex\2026-10-29\problem-statement-using-rag-a-college\outputs\college_large_dataset"
)

@st.cache_resource
def load_vectorstore():
    # Load every .txt file in the dataset folder
    txt_files = sorted(DATA_DIR.glob("*.txt"))

    if len(txt_files) < 5:
        raise FileNotFoundError(
            f"Expected at least 5 .txt files in {DATA_DIR}, "
            f"but found {len(txt_files)}."
        )

    documents = []
    for file_path in txt_files:
        loaded_docs = TextLoader(
            str(file_path),
            encoding="utf-8"
        ).load()

        # Keep the filename for source citations
        for document in loaded_docs:
            document.metadata["source"] = file_path.name

        documents.extend(loaded_docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )
    final_documents = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(final_documents, embeddings)


# Load and index the files when the app starts
db = load_vectorstore()

st.success("Text files loaded from the dataset folder.")

#load vector DB (ONLY ONCE)
db = load_vectorstore()

query = st.text_input("Enter your question here")

if query:
    docs = db.similarity_search(query, k=3)
    st.subheader("Retrived Context:")
    for i, doc in enumerate(docs):
        st.markdown(f"**Result{i+1}:**")
        st.write(doc.page_content)