from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

racine_projet = Path(__file__).parent.parent
source_db = racine_projet / "chroma_db"

if not source_db.exists():
    print(f"Impossible de trouver la base à cet endroit {source_db}")
else : 
    embedding_model = OllamaEmbeddings(model="llama3")
    llm = ChatOllama(model="llama3")
    vector_store = Chroma(
        embedding_function=embedding_model,
        persist_directory=str(source_db),
    )

    question = input('Quelle est la question ? ')
    results = vector_store.similarity_search(
        question,
        k=2
    )
    contexte_trouve = ""
    for index, res in enumerate(results, 1):
        # print(f"resultat {index}\n")
        # print(f"texte : {res.page_content}\n")
        # print(f"metadonnéees : {res.metadata}\n")

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
    
    print("\n Génération de la réponse")
    
    
    reponse_ia = llm.invoke(prompt_final)
    
    print("\n REPONSE FINALE :")
    print(reponse_ia.content)

    