import streamlit as st
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama


st.set_page_config(page_title="RAG Local")
st.title("Projet RAG")
st.write("Posez une question sur les documents internes.")

  
@st.cache_resource
def charger_moteur():
    racine_projet = Path(__file__).parent
    source_db = racine_projet / "chroma_db"
    
    if not source_db.exists():
        return None, None
        
    embedding_model = OllamaEmbeddings(model="llama3")
    llm = ChatOllama(model="llama3")
    vector_store = Chroma(
        embedding_function=embedding_model,
        persist_directory=str(source_db),
    )
    return vector_store, llm

vector_store, llm = charger_moteur()

if not vector_store:
    st.error("Base de données introuvable. Veuillez lancer l'ingestion d'abord.")
    st.stop() 

question = st.chat_input("Ex: Le PDF est-il interactif ?")

if question:
    st.chat_message("user").write(question)
    
    with st.spinner("Recherche dans les documents..."):
        
        results = vector_store.similarity_search(question, k=2)
        
        contexte_trouve = ""
        for res in results:
            contexte_trouve += res.page_content + "\n\n"
            
        prompt_final = f"""
        Tu es un assistant expert de chez Thales. 
        Réponds à la question de l'utilisateur EN TE BASANT UNIQUEMENT sur le contexte fourni ci-dessous.
        Si la réponse n'est pas dans le contexte, dis "Je ne sais pas".
        
        Contexte :
        {contexte_trouve}
        
        Question : {question}
        
        Réponse :
        """
        
        reponse_ia = llm.invoke(prompt_final)
        
        st.chat_message("assistant").write(reponse_ia.content)