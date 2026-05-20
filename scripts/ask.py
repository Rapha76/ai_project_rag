from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
import sys

racine_projet = Path(__file__).parent.parent
source_db = racine_projet / "chroma_db"

if not source_db.exists():
    print(f"Impossible de trouver la base à cet endroit {source_db}")
else : 
    # --- 1. LES NOUVEAUX MODÈLES ---
    embedding_model = OllamaEmbeddings(model="nomic-embed-text", base_url="http://host.docker.internal:11434")
    llm = ChatOllama(model="llama3.2:1b", base_url="http://host.docker.internal:11434")
    
    vector_store = Chroma(
        embedding_function=embedding_model,
        persist_directory=str(source_db),
    )

    question = input('\nQuelle est la question ? ')
    
    # On garde k=2 pour la vitesse
    results = vector_store.similarity_search(question, k=2)
    
    contexte_trouve = ""
    for index, res in enumerate(results, 1):
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
    
    print("\nGénération de la réponse...\n")
    
    # --- 2. L'EFFET MACHINE À ÉCRIRE DANS LE TERMINAL ---
    for chunk in llm.stream(prompt_final):
        # On affiche chaque morceau de texte au fur et à mesure sans sauter de ligne
        print(chunk.content, end="", flush=True)
    
    print("\n")