BASE_DIR = "/Users/victormontaluisa/Desktop/LOCAL\ CAPACITACION\ 2024/LOCAL\ KEEPCODING\ IA/ENTREGA_FINAL/REPOSITORIO_DOCUMENTOS/repositorio_normas/"


import os
import json
import shutil
import time
from datetime import datetime
import fitz  # PyMuPDF para leer PDFs

# Directorios base
NEW_DIR = os.path.join(BASE_DIR, "01_PDF_NUEVOS")
PROCESSED_DIR = os.path.join(BASE_DIR, "02_PDF_PROCESADOS")
ERROR_DIR = os.path.join(BASE_DIR, "03_PDF_ERRORES")
METADATA_DIR = os.path.join(BASE_DIR, "02_JSON_METADATA")
OCR_DIR = os.path.join(BASE_DIR, "03_OCR")
LOG_FILE = os.path.join(BASE_DIR, "00_LOGS", "repository_log.txt")
INDEX_FILE = os.path.join(BASE_DIR, "index.json")

# Asegurar que los directorios existen
for folder in [NEW_DIR, PROCESSED_DIR, ERROR_DIR, METADATA_DIR, OCR_DIR, os.path.join(BASE_DIR, "00_LOGS")]:
    os.makedirs(folder, exist_ok=True)

def log_action(action):
    """Registrar acciones en un archivo de log."""
    with open(LOG_FILE, "a") as log:
        log.write(f"{datetime.now().isoformat()} - {action}\n")

def extract_metadata(pdf_path):
    """Extraer metadatos básicos de un PDF."""
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        num_pages = len(doc)
        size = os.path.getsize(pdf_path)
        
        return {
            "filename": os.path.basename(pdf_path),
            "size_kb": round(size / 1024, 2),
            "num_pages": num_pages,
            "title": metadata.get("title", "Unknown"),
            "author": metadata.get("author", "Unknown"),
            "creation_date": metadata.get("creationDate", "Unknown"),
        }
    except Exception as e:
        log_action(f"Error al extraer metadatos de {pdf_path}: {e}")
        return None

def update_index():
    """Actualizar el archivo index.json con los documentos procesados."""
    index_data = []
    
    for filename in os.listdir(PROCESSED_DIR):
        if filename.endswith(".pdf"):
            metadata_path = os.path.join(METADATA_DIR, f"{filename}.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    index_data.append(metadata)
    
    with open(INDEX_FILE, "w") as f:
        json.dump(index_data, f, indent=4)

def process_new_pdfs():
    """Mover PDFs de la carpeta nuevos a procesados y extraer metadatos."""
    for filename in os.listdir(NEW_DIR):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(NEW_DIR, filename)
            metadata = extract_metadata(pdf_path)
            
            if metadata:
                # Guardar metadatos en un archivo JSON
                metadata_path = os.path.join(METADATA_DIR, f"{filename}.json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=4)
                
                # Mover el archivo a procesados
                shutil.move(pdf_path, os.path.join(PROCESSED_DIR, filename))
                log_action(f"Procesado y movido: {filename}")
            else:
                # Mover el archivo a errores
                shutil.move(pdf_path, os.path.join(ERROR_DIR, filename))
                log_action(f"Error al procesar: {filename}")

    update_index()
    log_action("Índice de documentos actualizado.")

# Ejecutar procesamiento
if __name__ == "__main__":
    print("Procesando documentos nuevos...")
    process_new_pdfs()
    print("Proceso completado.")
