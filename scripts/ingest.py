from docling.document_converter import DocumentConverter
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


racine_projet = Path(__file__).parent.parent
source = racine_projet / "data" / "pdf-exemple.pdf"

if not source.exists():
    print(f"Impossible de trouver le fichier à cet endroit {source}")
else : 
    print(f"Fichier trouvé : {source.name}")
    try : 
        converter = DocumentConverter()
        doc = converter.convert(source).document
        text_markdown = doc.export_to_markdown()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200
        )
        chunks = text_splitter.split_text(text_markdown)

        embedding_model = OllamaEmbeddings(model="llama3")
        vector_chroma = Chroma.from_texts(
            texts = chunks,
            embedding= embedding_model,
            persist_directory= str(racine_projet / "chroma_db")
        )

        print(f"{len(chunks)} morceaux de texte sauvegardé dans la base vectorielle.")

    except Exception as e : 
        print(f"erreur pendant la conversion {e}")