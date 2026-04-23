from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

racine_projet = Path(__file__).parent.parent
source_db = racine_projet / "chroma_db"

if not source_db.exists():
    print(f"Impossible de trouver la base à cet endroit {source_db}")
else : 
    embedding_model = OllamaEmbeddings(model="llama3")
    vector_store = Chroma(
        embedding_function=embedding_model,
        persist_directory=str(source_db),
    )
    question = "PDF est-il statique ?"
    results = vector_store.similarity_search(
        question,
        k=2
    )
    for index, res in enumerate(results, 1):
        print(f"resultat {index}\n")
        print(f"texte : {res.page_content}\n")
        print(f"metadonnéees : {res.metadata}\n")