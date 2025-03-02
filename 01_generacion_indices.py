import os
import shutil
import fitz  # PyMuPDF
import pytesseract
import json
import pymongo
import logging
import re
import uuid
import faiss
import numpy as np
from pdf2image import convert_from_path
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Query
from typing import List
import pytz
from datetime import datetime

import spacy
from spacy.lang.es.stop_words import STOP_WORDS
import urllib.parse
from tqdm import tqdm
from openai import OpenAI
import torch
import logging
import traceback

import chromadb

# Cargar configuraciones desde .env
load_dotenv()


##############################################################################
#
# REPOSITORIO DE PROCESAMIENTO DE DOCUMENTOS
#
#https://github.com/vmontaluisa/003_CONVERSION_OCR_preprocesamiento
BASE_DIR = os.getenv("BASE_DIR", "../REPOSITORIO_DOCUMENTOS/preprocesamiento")
##############################################################################


DIR_NUEVOS = os.path.join(BASE_DIR, os.getenv("DIR_NUEVOS", "01_PDF_NUEVOS"))
DIR_PROCESADOS = os.path.join(BASE_DIR, os.getenv("DIR_PROCESADOS", "02_PDF_PROCESADOS"))
DIR_ERRORES = os.path.join(BASE_DIR, os.getenv("DIR_ERRORES", "03_PDF_ERRORES"))
DIR_METADATA = os.path.join(BASE_DIR, os.getenv("DIR_METADATA", "04_JSON_METADATA"))
DIR_OCR = os.path.join(BASE_DIR, os.getenv("DIR_OCR", "05_OCR"))
DIR_IMAGENES = os.path.join(BASE_DIR, os.getenv("DIR_IMAGENES", "06_IMAGENES"))
DIR_LOGS = os.path.join(BASE_DIR, os.getenv("DIR_LOGS", "00_LOGS"))
DIR_FAISS = os.path.join(BASE_DIR, os.getenv("DIR_FAISS", "07_FAISS"))
DIR_CHROMA = os.path.join(BASE_DIR, os.getenv("DIR_CHROMA", "07_CHROMADB"))



# Conectar a MongoDB usando variables de entorno
#MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "documentos_legales")
MONGODB_HOST=os.getenv("MONGODB_HOST","")
MONGODB_PORT=os.getenv("MONGODB_PORT","")
MONGODB_AUTH_SOURCE=os.getenv("MONGODB_AUTH_SOURCE","")
MONGODB_USER=os.getenv("MONGODB_USER","")
MONGODB_PASSWORD=os.getenv("MONGODB_PASSWORD","")
MONGO_COLLECTION_CHROMA = os.getenv("MONGO_COLLECTION_CHROMA", "")
MONGO_COLLECTION_FAISS = os.getenv("MONGO_COLLECTION_FAISS", "")
MONGO_COLLECTION_PARRAFOS = os.getenv("MONGO_COLLECTION_PARRAFOS", "")
MONGODB_USER_ENCODED = urllib.parse.quote_plus(MONGODB_USER)
MONGODB_PASSWORD_ENCODED = urllib.parse.quote_plus(MONGODB_PASSWORD)
MONGODB_URI = f"mongodb://{MONGODB_USER_ENCODED}:{MONGODB_PASSWORD_ENCODED}@{MONGODB_HOST}:{MONGODB_PORT}/{MONGODB_PORT}?authSource={MONGODB_AUTH_SOURCE}"

#Open
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", "")
OPENAI_MODELO=os.getenv("OPENAI_MODELO", "")
LENGUAJE=os.getenv("LENGUAJE", "")
OPENAI_MAXIMO_TEXTO=int(os.getenv("OPENAI_MAXIMO_TEXTO", ""))

#CHROMA_DB
CHROMA_DB_PATH=f"{DIR_CHROMA}"  # Ruta donde se guardará la base de datos
os.makedirs(CHROMA_DB_PATH, exist_ok=True)


# Pregunta OPENAI
PREGUNTAS = ["¿Numero Registro Oficial o Suplemento?",
                 "¿Fecha de publicacion en formato YYYY-MM-DD?", 
                 "¿Cual es el titulo de la Ley?",
                
                 ]

# Diccionario de términos legales personalizados
TERMINOS_LEGALES = {
    "vicio de consentimiento": "error esencial",
    "acción de nulidad": "recurso de nulidad",
    "resolución judicial": "sentencia",
   
}
# Lista de términos reemplazables
TERMINOS_REEMPLAZABLES = [
    {"termino_original": "art. ", "termino_final": "artículo. "},
    {"termino_original": "Considerandos ", "termino_final": "Considerando "}
]

#lsitado de terminos para olvidar parrafgos en MINUSCULAS preprocesado
TERMINOS_NO_CONSIDERAR_PARRAFOS = {
    'lexis',
    'jurisxx',
    'concordancias:'
}

#lsitado de terminos para olvidar parrafgos en MINUSCULAS postprocesado
PALABRAS_IGNORADAS = { 
                      "concordancia",
                      
                      }  # Palabras poco útiles


##############################################################################


# Crear carpetas si no existen
for d in [DIR_NUEVOS, DIR_PROCESADOS, DIR_ERRORES, DIR_METADATA, DIR_OCR, DIR_IMAGENES, DIR_LOGS, DIR_FAISS]:
    os.makedirs(d, exist_ok=True)

#coneccion mongo ##########################################
client = pymongo.MongoClient(MONGODB_URI)
db = client[MONGO_DB]
coleccion_chroma = db[MONGO_COLLECTION_CHROMA]
coleccion_faiss = db[MONGO_COLLECTION_FAISS]
coleccion_documentos_parrafos = db[MONGO_COLLECTION_PARRAFOS]


# Definir zona horaria de Guayaquil###########################################
zona_horaria = pytz.timezone("America/Guayaquil")

def guayaquil_time(*args):
    """Convierte la hora UTC a la zona horaria de Guayaquil."""
    utc_dt = datetime.utcnow().replace(tzinfo=pytz.utc)
    local_dt = utc_dt.astimezone(zona_horaria)
    return local_dt.timetuple()
# Configurar logging
log_file = os.path.join(DIR_LOGS, "procesamiento.log")
logging.Formatter.converter = guayaquil_time  # Establecer la conversión a Guayaquil
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)



#######################################################################################
#######################################################################################

chroma_client=None
chroma_collection=None

# Inicializar cliente de ChromaDB
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = chroma_client.get_or_create_collection(name="documentos_legales")
    logging.info(f"✅ Base de datos ChromaDB creada en {CHROMA_DB_PATH}")
except Exception as e:
    logging.error(f"❌ Error iniciando ChromaDB: {e}\n{traceback.format_exc()}")



nlp = spacy.load("es_core_news_sm")

# Verificar si MPS está disponible en Mac
device = "mps" if torch.backends.mps.is_available() else "cpu"
# Cargar el modelo en MPS

os.environ["TOKENIZERS_PARALLELISM"] = "false"
#MODELO="PlanTL-GOB-ES/roberta-base-bne"
MODELO="hiiamsid/sentence_similarity_spanish_es"

modelo_legal = SentenceTransformer(MODELO, device=device)
logging.info(f"✅ Modelo cargado en: {device}")

# Crear índice FAISS para embeddings
embedding_dim = 768
INDEX_FILE = os.path.join(DIR_FAISS, "faiss_index.idx")

# Iniciar ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
chroma_collection = chroma_client.get_or_create_collection(name="documentos_legales")



#######################################################################################
#######################################################################################

def limpiar_json(datos):
    """Reemplaza valores None con una cadena vacía ("")."""
    
    if isinstance(datos, dict):
        return {k: limpiar_json(v) for k, v in datos.items()}
    elif isinstance(datos, list):
        return [limpiar_json(v) for v in datos]
    elif datos is None:
        return ""
    else:
        return datos



def agregar_documento_chroma(texto, metadata):##############################################
    """Genera embeddings y los almacena en ChromaDB con metadatos."""
    try:
        embedding = modelo_legal.encode([texto] , 
                                        show_progress_bar=False ,
                                         normalize_embeddings=True
                                        )[0].tolist()
        if metadata and "_id" in metadata:
            del metadata["_id"]

        metadata = limpiar_json(metadata)  # Asegurar que los metadatos no tengan None

        # Asegurar que el ID está presente
        if "vector_id" not in metadata:
            metadata["vector_id"] = str(uuid.uuid4())

        # Agregar a ChromaDB
        chroma_collection.add(
            ids=[metadata["vector_id"]],
            embeddings=[embedding],
            metadatas=[metadata]
        )
      #  logging.info(f"✅ Documento agregado a ChromaDB con ID: {metadata['vector_id']}")
    
    except Exception as e:
        logging.error(f"❌ Error al agregar a ChromaDB: {e}\n{traceback.format_exc()}")



#######################################################################################
#######################################################################################

def cargar_indice():
    """Carga el índice FAISS si existe, o crea uno nuevo."""
    global index
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
    else:
        index = faiss.IndexFlatL2(embedding_dim)        
        logging.info("🔄 Creando un nuevo índice FAISS vacío." )

def guardar_indice():
    """Guarda el índice FAISS en disco."""
    faiss.write_index(index, INDEX_FILE)

def agregar_documento_faiss(texto, metadata_json):###########################################
    """Genera embeddings para el texto y los agrega al índice FAISS."""
    embedding = modelo_legal.encode([texto], 
                                    show_progress_bar=False ,
                                     normalize_embeddings=True
                                    )[0]
    index.add(np.array([embedding]).astype('float32'))
    
    guardar_indice()  # Guardamos el índice actualizado
    
    ultimo_indice=index.ntotal - 1
    
    metadata_json['indice_faiss']=ultimo_indice
    coleccion_faiss.insert_one(metadata_json)

    
    
        # Retornar el índice asignado al documento en FAISS
    return ultimo_indice  # El último índice agregado
#######################################################################################
#######################################################################################
 

def preprocesar_texto(texto):
    """Aplica lematización y elimina stopwords del texto."""
    doc = nlp(texto)
    palabras_filtradas = [
        token.lemma_ for token in doc 
        if token.text.lower() not in STOP_WORDS and not token.is_punct
    ]
    return " ".join(palabras_filtradas)


def expandir_texto(texto):
    """Agrega sinónimos legales al texto para mejorar embeddings."""
    for term, sinonimo in TERMINOS_LEGALES.items():
        if term in texto:
            texto += f" {sinonimo}"
    return texto

def reemplazar_terminos(texto): 
    """
    Reemplaza términos en un texto según una lista de diccionarios con 'termino_original' y 'termino_final'.      """
    for reemplazo in TERMINOS_REEMPLAZABLES:
        texto = texto.replace(reemplazo["termino_original"], reemplazo["termino_final"])
    return texto


def clean_text(text):
    """Limpia el texto eliminando caracteres innecesarios y normalizando espacios."""
    text = re.sub(r'[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ0-9,. ]+', ' ', text)  # Solo caracteres válidos
    text = re.sub(r'\s+', ' ', text).strip()  # Eliminar espacios extras
    return text.lower()

def tiene_texto(pdf_path):
    """Determina si el PDF tiene texto seleccionable."""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            if page.get_text():
                return True
        return False
    except Exception as e:
        logging.error(f"Error verificando texto en {pdf_path}: {e}\n{traceback.format_exc()}")
        return False
    
def no_contiene_termino_prohibido(texto):
    """
    Verifica si ningún término en terminos_eliminar está presente en el texto.
     """
    texto = texto.lower()  # Convertir a minúsculas para búsqueda sin distinción de mayúsculas
    if texto.lower() in TERMINOS_NO_CONSIDERAR_PARRAFOS:
        return False
    return True
 
    
 

def es_texto_relevante_post_procesado(texto):
    """Filtra textos cortos o irrelevantes despeus de indexzarlso de indexarlos."""
    palabras = texto.split()
    if len(palabras) < 4:  # Si tiene menos de 4 palabras, es irrelevante
        return False
    if texto.lower() in PALABRAS_IGNORADAS:
        return False
    return True


def tiene_texto(pdf_path):
    """Determina si el PDF tiene texto seleccionable."""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            if page.get_text():
                return True
        return False
    except Exception as e:
        logging.error(f"Error verificando texto en {pdf_path}: {e}\n{traceback.format_exc()}")
        return False
    
    
    
def extract_text_with_ocr(pdf_path):
    """Extrae texto de un PDF escaneado usando OCR con Tesseract."""
    try:
        images = convert_from_path(pdf_path)
        extracted_text = ""
        for img in images:
            text = pytesseract.image_to_string(img , lang=LENGUAJE )
            extracted_text += text + "\n"
        return extracted_text.strip()
    except Exception as e:
        logging.error(f"Error al realizar OCR en {pdf_path}: {e}\n{traceback.format_exc()}")


def extract_metadata_txt_with_openai(text, questions):
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            text=text[:OPENAI_MAXIMO_TEXTO]
            prompt = ("""
            Extrae la siguiente información del texto de manera estricta, sin inferencias ni interpretaciones. 
            Si la información no está en el documento, responde 'No encontrado'. Devuelve la información en formato JSON.
            """ + "\n".join([f"- {q}" for q in questions]) + "\n\nTexto:\n" + text)
            
            response = client.chat.completions.create(
                model=OPENAI_MODELO,
                messages=[{"role": "system", "content": "Eres un asistente que extrae información clave exclusivamente del texto proporcionado. No hagas suposiciones. Devuelve siempre una respuesta en formato JSON."},
                        {"role": "user", "content": prompt}],
                temperature=0.0
            )
            
            metadata_json = json.loads(response.choices[0].message.content.strip())
            return metadata_json
        except Exception as e:
            logging.error(f"Error al procesar metadata con OpenAI en  {text}: {e}\n{traceback.format_exc()}")
            return {}
        
def extract_metadata_and_text(pdf_path):
    """Extrae metadatos y texto de un PDF, usando OCR si es necesario."""
    try:
        text = extract_text_from_pdf(pdf_path)
        
         # Guardar el texto extraído en un archivo .txt
        txt_path = os.path.join(DIR_OCR, f"{os.path.splitext(os.path.basename(pdf_path))[0]}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        
    
    except Exception as e:
        logging.error(f"Error al extract_metadata_and_text  {pdf_path}: {e}\n{traceback.format_exc()}")
        return None


def extract_text_from_pdf(pdf_path):
    """Extrae texto de un PDF si tiene contenido seleccionable."""
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return text.strip()
    except Exception as e:
        logging.error(f"Error al extraer texto de  {pdf_path}: {e}\n{traceback.format_exc()}")
        return None    

def procesar_pdfs():########################################################################################################
    """Procesa todos los PDFs en la carpeta de nuevos documentos."""
    archivos_pdf = [f for f in os.listdir(DIR_NUEVOS) if f.endswith(".pdf")]

    if not archivos_pdf:
        logging.info("📂 No hay nuevos PDFs para procesar.")
        return

    for archivo in  tqdm(archivos_pdf, desc=f"Proceso Total PDFs"  , disable=(len(archivos_pdf) == 1) ):
        pdf_path = os.path.join(DIR_NUEVOS, archivo)
        logging.info(f"🔍 Procesando: {archivo}")
        
        if tiene_texto(pdf_path):
            try:
                if extraer_texto_y_metadatos(pdf_path,archivo):
                    shutil.move(pdf_path, os.path.join(DIR_PROCESADOS, archivo))
                    logging.info(f"✅ Procesado correctamente: {archivo}")
                else:
                    shutil.move(pdf_path, os.path.join(DIR_ERRORES, archivo))
                    logging.error(f"❌ Error al procesar: {archivo}")
            except Exception as e:
                shutil.move(pdf_path, os.path.join(DIR_ERRORES, archivo))
                logging.error(f"❌ Error procesando {archivo}: {e}\n{traceback.format_exc()}")
        else:
            shutil.move(pdf_path, os.path.join(DIR_ERRORES, archivo))
            logging.error(f"❌ no hay texto {archivo}: {e}\n{traceback.format_exc()}")


def extraer_texto_y_metadatos(pdf_path,nomnre_archivo):#################################################################################
    """Extrae el texto del PDF, genera embeddings y almacena en FAISS."""
    try:
        doc = fitz.open(pdf_path)
        
        metadata = doc.metadata
        num_pages = len(doc)
        size = os.path.getsize(pdf_path)
        

        unid_archivo=str(uuid.uuid4())
        metadatos = {
            "archivo_nombre": nomnre_archivo,
            "archivo_nombre_espacios": os.path.basename(pdf_path).replace(" ", "_"),
            "archivo_titulo": metadata.get("title", "Unknown"),
            "archivo_unid": unid_archivo,
            "archivo_size_kb": round(size / 1024, 2),
            "archivo_author": metadata.get("author", "Unknown"),
            "archivo_creation_date": metadata.get("creationDate", "Unknown"),
            "archivo_paginas": num_pages,
            "texto": []
        }
        extract_metadata_and_text(pdf_path) #Genera el documento ocrtxt

        articulo_actual = None
        imagenes_paginas = convert_from_path(pdf_path, fmt='png')
        numero_pagina=0
        nombre_seccion=''
        metadata_extra={}
        metadata_adicional={}
        for num_pagina in  tqdm(range(len(doc)), desc=f"Procesando páginas del PDF[{nomnre_archivo}]"):
            numero_pagina+=1
            pagina = doc[num_pagina]
            parrafos = pagina.get_text("blocks")
            if numero_pagina==1:#consutal de metadatas solo con la primera pagina
                texto_pagina=pagina.get_text("text")
                metadata_adicional = {}#extract_metadata_txt_with_openai(clean_text(texto_pagina), PREGUNTAS)

            
            
            # Guardar imagen de la página
            nommbre_imagen=f"{metadatos['archivo_unid']}_pag{num_pagina+1}.png"
            img_filename = os.path.join(DIR_IMAGENES,nommbre_imagen )
            imagenes_paginas[num_pagina].save(img_filename, "PNG")
            numero_parrafos=0
            for bloque in parrafos:
                texto_original=bloque[4].strip()
                if   no_contiene_termino_prohibido(texto_original):                
                    numero_parrafos+=1
                    ################################################################
                    ##IDENTIFICACION DE SECCIONES

                    texto_mayusculas=texto_original.upper()
                    
                    pattern = r"^CONSIDERANDO"
                    match = re.match(pattern, texto_mayusculas, re.IGNORECASE)
                    if match:    
                            nombre_seccion=clean_text(texto_original)
                            nombre_seccion=nombre_seccion.upper()
                            articulo_actual=''
                                        
                    pattern = r"^DISPOSICIONES\s+GENERALES"
                    match = re.match(pattern, texto_mayusculas, re.IGNORECASE)
                    if match:    
                            nombre_seccion=clean_text(texto_original)
                            nombre_seccion=nombre_seccion.upper()
                            articulo_actual=''
                    
                    pattern = r"^DISPOSICIONES\s+DEROGATORIAS"
                    match = re.match(pattern, texto_mayusculas, re.IGNORECASE)
                    if match:    
                            nombre_seccion=clean_text(texto_original)
                            nombre_seccion=nombre_seccion.upper()
                            articulo_actual=''

                    pattern = r"^DISPOSICI[ÒO]N\s+DEROGATORIA"
                    match = re.match(pattern, texto_mayusculas, re.IGNORECASE)
                    if match:    
                            nombre_seccion=clean_text(texto_original)
                            nombre_seccion=nombre_seccion.upper()
                            articulo_actual=''

                            
                    pattern = r"^DEROGATORIAS"
                    match = re.match(pattern, texto_mayusculas, re.IGNORECASE)
                    if match:    
                            nombre_seccion=clean_text(texto_original)
                            nombre_seccion=nombre_seccion.upper()
                            articulo_actual=''
                            

                    
                    pattern = r"^DISPOSICIONES\s+REFORMATORIAS"
                    match = re.match(pattern, texto_mayusculas, re.IGNORECASE)
                    if match:    
                            nombre_seccion=clean_text(texto_original)
                            nombre_seccion=nombre_seccion.upper()
                            articulo_actual=''

                    
                    pattern = r"^DISPOSICIONES\s+TRANSITORIAS"
                    match = re.match(pattern, texto_mayusculas, re.IGNORECASE)
                    if match:    
                            nombre_seccion=clean_text(texto_original)
                            nombre_seccion=nombre_seccion.upper()
                            articulo_actual=''

                            
                    pattern = r"^\s*T[ÍI]TULO\b"
                    match = re.match(pattern, texto_mayusculas, re.IGNORECASE)
                    if match:    
                            nombre_seccion=clean_text(texto_original)
                            nombre_seccion=nombre_seccion.upper()
                            articulo_actual=''

                    pattern = r"^\s*CAP[ÍI]TULO\b"
                    match = re.match(pattern, texto_mayusculas, re.IGNORECASE)
                    if match:    
                            nombre_seccion=clean_text(texto_original)
                            nombre_seccion=nombre_seccion.upper()
                            articulo_actual=''


                    pattern = r"^\s*SECCI[ÓO]N\b"
                    match = re.match(pattern, texto_mayusculas, re.IGNORECASE)
                    if match:    
                            nombre_seccion=clean_text(texto_original)
                            nombre_seccion=nombre_seccion.upper()
                            articulo_actual=''

                    pattern = r"^DISPOSICI[ÓO]N\s+FINAL"
                    match = re.match(pattern, texto_mayusculas, re.IGNORECASE)
                    if match:    
                            nombre_seccion=clean_text(texto_original)
                            nombre_seccion=nombre_seccion.upper()
                            articulo_actual=''

                    pattern = r"ART\.\s*([\d\.]+)\.-"                  
                    match = re.match(pattern, texto_mayusculas, re.IGNORECASE)
                    if match:    
                             articulo_actual = match.group(1)

                    pattern = r"ART[ÍI]CULO\s+(\d+)\."
                    match = re.match(pattern, texto_mayusculas, re.IGNORECASE)
                    if match:    
                             articulo_actual = match.group(1)
                            
                    ################################################################
                            
                    
                    texto_parrafo = clean_text(texto_original) #caaracters especiales y  quitar  mayusculas
                    texto_parrafo = expandir_texto(texto_parrafo)   #reemplazar con sinonimos
                    texto_parrafo = reemplazar_terminos(texto_parrafo) #reemplazo
                    texto_parrafo = preprocesar_texto(texto_parrafo)  # Aplicar lematización y stopwords

                    if texto_parrafo:
                        if es_texto_relevante_post_procesado(texto_parrafo):
                            coordenadas = {
                                "x0": bloque[0], "y0": bloque[1], "x1": bloque[2], "y1": bloque[3]
                            }                        
                            unid_parrafo = str(uuid.uuid4())                        
                            metadatos["texto"].append({
                                "archivo_uuid":unid_archivo,
                                "archivo_nombre_archivo_pdf":nomnre_archivo,
                                "pagina_numero":numero_pagina,
                                "pagina_imagen": nommbre_imagen,
                                "seccion_nombre":nombre_seccion,
                                "articulo_numero": articulo_actual,
                                "parrafo_unid": unid_parrafo,
                                "parrafo_numero":numero_parrafos,
                                "parrafo_coordenadas": coordenadas,
                                "parrafo_texto_indexado": texto_parrafo,
                                "parrafo_texto_original": texto_original,

                            })
                            metadata_indice= {
                                "vector_id": unid_parrafo,
                                "archivo_uuid":unid_archivo,
                                "archivo_nombre_archivo_pdf":nomnre_archivo,
                                "pagina_numero":numero_pagina,                            
                                "pagina_imagen": nommbre_imagen,
                                "seccion_nombre":nombre_seccion,
                                "articulo_numero": articulo_actual,
                                "parrafo_unid": unid_parrafo,
                                "parrafo_numero":numero_parrafos,
                                "parrafo_coordenadas_x0":(coordenadas["x0"]),
                                "parrafo_coordenadas_y0":(coordenadas["y0"]),
                                "parrafo_coordenadas_x1":(coordenadas["x1"]),
                                "parrafo_coordenadas_y1":(coordenadas["y1"]),
                                "parrafo_texto_indexado": texto_parrafo,
                                "parrafo_texto_original": texto_original,
                                                        
                            }
                                                        
                            agregar_documento_faiss(texto_parrafo,metadata_indice)
                            # Agregar a ChromaDB
                            agregar_documento_chroma(texto_parrafo, metadata_indice)


   
        
        metadatos.update(metadata_adicional)
        json_path = os.path.join(DIR_METADATA, os.path.basename(pdf_path).replace(".pdf", "_metadata.json").replace(" ", "_"))
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadatos, f, indent=4, ensure_ascii=False)

        coleccion_documentos_parrafos.insert_one(metadatos)

        logging.info(f"✅ Texto extraído y almacenado: {pdf_path}")
        return True
    except Exception as e:
        logging.error(f"❌ Error extrayendo texto de {pdf_path}: {e}\n{traceback.format_exc()}")
        return False


##############################################################################################################

if __name__ == "__main__":
    logging.info("🚀 Iniciando procesamiento de PDFs...")
    cargar_indice()
    procesar_pdfs()
    logging.info("✅ Procesamiento finalizado.")
    
    
    
   # cargar_indice()

