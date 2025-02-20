BASE_DIR = "../REPOSITORIO_DOCUMENTOS/repositorio_normas/"

import os
import json
import shutil
import time
from datetime import datetime
import fitz  # PyMuPDF para leer PDFs
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

# Directorios base
#BASE_DIR = "/Users/tu_usuario/Documents/RepositorioPDFs/"
NEW_DIR = os.path.join(BASE_DIR, "01_PDF_NUEVOS")
PROCESSED_DIR = os.path.join(BASE_DIR, "02_PDF_PROCESADOS")
ERROR_DIR = os.path.join(BASE_DIR, "03_PDF_ERRORES")
METADATA_DIR = os.path.join(BASE_DIR, "02_JSON_METADATA")
OCR_DIR = os.path.join(BASE_DIR, "03_OCR")
LOG_FILE = os.path.join(BASE_DIR, "00_LOGS", "repository_log.txt")
INDEX_FILE = os.path.join(BASE_DIR, "index.json")


# Asegurar que los directorios existen
#for folder in [NEW_DIR, PROCESSED_DIR, ERROR_DIR, METADATA_DIR, OCR_DIR, os.path.join(BASE_DIR, "00_LOGS")]:
#    os.makedirs(folder, exist_ok=True)

def log_action(action):
    """Registrar acciones en un archivo de log."""
    with open(LOG_FILE, "a") as log:
        log.write(f"{datetime.now().isoformat()} - {action}\n")

def extract_text_from_pdf(pdf_path):
    """Extrae texto de un PDF si tiene contenido seleccionable."""
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return text.strip()
    except Exception as e:
        log_action(f"Error al extraer texto de {pdf_path}: {e}")
        return None

def extract_text_with_ocr(pdf_path):
    """Extrae texto de un PDF escaneado usando OCR con Tesseract."""
    try:
        images = convert_from_path(pdf_path)
        extracted_text = ""
        for img in images:
            text = pytesseract.image_to_string(img)
            extracted_text += text + "\n"
        return extracted_text.strip()
    except Exception as e:
        log_action(f"Error al realizar OCR en {pdf_path}: {e}")
        return None

def extract_metadata_and_text(pdf_path):
    """Extrae metadatos y texto de un PDF, usando OCR si es necesario."""
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        num_pages = len(doc)
        size = os.path.getsize(pdf_path)
        text = extract_text_from_pdf(pdf_path)
        
        if not text:
            log_action(f"El documento {pdf_path} no tiene texto seleccionable. Aplicando OCR...")
            text = extract_text_with_ocr(pdf_path)
            if text:
                shutil.move(pdf_path, os.path.join(OCR_DIR, os.path.basename(pdf_path)))
            else:
                shutil.move(pdf_path, os.path.join(ERROR_DIR, os.path.basename(pdf_path)))
                log_action(f"Error: No se pudo extraer texto ni con OCR en {pdf_path}.")
                return None
        else:
            shutil.move(pdf_path, os.path.join(PROCESSED_DIR, os.path.basename(pdf_path)))
        
                # Guardar el texto extraído en un archivo .txt
        txt_path = os.path.join(OCR_DIR, f"{os.path.splitext(os.path.basename(pdf_path))[0]}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        return {
            "filename": os.path.basename(pdf_path),
            "size_kb": round(size / 1024, 2),
            "num_pages": num_pages,
            "title": metadata.get("title", "Unknown"),
            "author": metadata.get("author", "Unknown"),
            "creation_date": metadata.get("creationDate", "Unknown"),
            "extracted_text": text  # Guardar un fragmento del texto extraído
#            "extracted_text": text[:5000]  # Guardar un fragmento del texto extraído
        }
    except Exception as e:
        log_action(f"Error al procesar {pdf_path}: {e}")
        return None

def update_index():
    """Actualizar el archivo index.json con los documentos procesados."""
    index_data = []
    
    for filename in os.listdir(PROCESSED_DIR) + os.listdir(OCR_DIR):
        if filename.endswith(".pdf"):
            metadata_path = os.path.join(METADATA_DIR, f"{filename}.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    index_data.append(metadata)
    
    with open(INDEX_FILE, "w") as f:
        json.dump(index_data, f, indent=4)

def process_new_pdfs():
    """Procesar PDFs nuevos, aplicar OCR si es necesario y extraer metadatos."""
    for filename in os.listdir(NEW_DIR):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(NEW_DIR, filename)
            metadata = extract_metadata_and_text(pdf_path)
            
            if metadata:
                metadata_path = os.path.join(METADATA_DIR, f"{filename}.json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=4)
                    
               
                    
                    
                log_action(f"Procesado y guardado: {filename}")
            else:
                log_action(f"Error en procesamiento: {filename}")

    update_index()
    log_action("Índice de documentos actualizado.")

# Ejecutar procesamiento
if __name__ == "__main__":
    print("Procesando documentos nuevos...")
    process_new_pdfs()
    print("Proceso completado.")
