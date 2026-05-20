import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

OLLAMA_URL = "http://host.docker.internal:11434"
CHAT_MODEL = "llama3.2:1b"
EMBEDDING_MODEL = "nomic-embed-text"
DB_DIR = "chroma_db"

st.set_page_config(page_title="RAG Local", layout="wide")
st.title("Projet RAG")

@st.cache_resource
def get_vector_store():
    # Cette fonction ne s'exécutera qu'une seule fois !
    embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)
    os.makedirs(DB_DIR, exist_ok=True)
    return Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)

vector_store = get_vector_store()

with st.sidebar:
    st.header("Base de données")
    
    uploaded_files = st.file_uploader("Ajoutez vos PDF ici", type="pdf", accept_multiple_files=True)
    
    if st.button("Construction de la base de données") and uploaded_files:
        with st.spinner("Lecture des documents"):
            
            documents = []
            
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                loader = PyPDFLoader(tmp_file_path)
                documents.extend(loader.load())
                os.remove(tmp_file_path)
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            
            vector_store.add_documents(splits)
            
            st.success("Base de données construite")

try:
    db_is_ready = vector_store._collection.count() > 0
except:
    db_is_ready = False

if db_is_ready:
    st.success("La base de données est prête ! Posez vos questions.")
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    llm = ChatOllama(model=CHAT_MODEL, base_url=OLLAMA_URL)
    
    prompt = PromptTemplate(
        template="""Tu es un assistant utile. Utilise le contexte suivant pour répondre à la question. 
        Si tu ne connais pas la réponse, dis simplement que tu ne sais pas.
        
        Contexte : {context}
        
        Question : {input}
        
        Réponse : """,
        input_variables=["context", "input"]
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    question = st.chat_input("Posez votre question sur vos documents...")
    if question:
        st.chat_message("user").write(question)
        
        with st.chat_message("Réflexion"):
            st.write_stream(rag_chain.stream(question))

else:
    st.info("Bienvenue ! Veuillez ajouter des documents PDF dans la barre à gauche pour commencer.")