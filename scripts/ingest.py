from docling.document_converter import DocumentConverter
from pathlib import Path

racine_projet = Path(__file__).parent.parent
source = racine_projet / "data" / "doc1.pdf"

if not source.exists():
    print(f"Impossible de trouver le fichier à cet endroit {source}")
else : 
    print(f"Fichier trouvé : {source.name}")
    try : 
        converter = DocumentConverter()
        doc = converter.convert(source).document
        print(doc.export_to_markdown())
    except Exception as e : 
        print(f"erreur pendant la conversion {e}")