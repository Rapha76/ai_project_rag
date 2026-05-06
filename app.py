import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

OLLAMA_URL = "http://host.docker.internal:11434"
MODEL_NAME = "phi3"
DB_DIR = "chroma_db"

st.set_page_config(page_title="RAG Local", layout="wide")
st.title("Projet RAG")

with st.sidebar:
    st.header("Base de données")
    
    uploaded_files = st.file_uploader("Ajoutez vos PDF ici", type="pdf", accept_multiple_files=True)
    
    if st.button("Construction de la base de données") and uploaded_files:
        with st.spinner("Lecture des documents"):
            
            documents = []
            embedding_model = OllamaEmbeddings(model=MODEL_NAME, base_url=OLLAMA_URL)
            
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                loader = PyPDFLoader(tmp_file_path)
                documents.extend(loader.load())
                os.remove(tmp_file_path)
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            
            Chroma.from_documents(documents=splits, embedding=embedding_model, persist_directory=DB_DIR)
            
            st.success("Base de données construite")

if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
    st.success("La base de données est prête ! Posez vos questions.")
    
    embedding_model = OllamaEmbeddings(model=MODEL_NAME, base_url=OLLAMA_URL)
    vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_URL)
    
    prompt = PromptTemplate(
        template="""Tu es un assistant utile. Utilise le contexte suivant pour répondre à la question. 
        Si tu ne connais pas la réponse, dis simplement que tu ne sais pas.
        
        Contexte : {context}
        
        Question : {input}
        
        Réponse : """,
        input_variables=["context", "input"]
    )
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    question = st.chat_input("Posez votre question sur vos documents...")
    if question:
        st.chat_message("user").write(question)
        
        with st.spinner("Réflexion"):
            response = retrieval_chain.invoke({"input": question})
            st.chat_message("assistant").write(response["answer"])

else:
    st.info("Bienvenue ! Veuillez ajouter des documents PDF dans la barre à gauche pour commencer.")