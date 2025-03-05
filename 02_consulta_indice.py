import os
import faiss
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import logging
import traceback
import torch
from fastapi import FastAPI, Query

import pytz
from datetime import datetime
import urllib.parse
import pymongo

from fastapi.middleware.cors import CORSMiddleware


# Cargar configuraciones desde .env
load_dotenv()



##############################################################################
#
# REPOSITORIO DE PROCESAMIENTO DE DOCUMENTOS
#  ../REPOSITORIO_DOCUMENTOS/preprocesamiento
#https://github.com/vmontaluisa/003_CONVERSION_OCR_preprocesamiento
BASE_DIR = os.getenv("BASE_DIR", "")
##############################################################################


MONGO_DB = os.getenv("MONGO_DB", "")
MONGODB_HOST=os.getenv("MONGODB_HOST","")
MONGODB_PORT=os.getenv("MONGODB_PORT","")
MONGODB_AUTH_SOURCE=os.getenv("MONGODB_AUTH_SOURCE","")
MONGODB_USER=os.getenv("MONGODB_USER","")
MONGODB_PASSWORD=os.getenv("MONGODB_PASSWORD","")
MONGO_COLLECTION_FAISS = os.getenv("MONGO_COLLECTION_FAISS", "")
MONGO_COLLECTION_PARRAFOS = os.getenv("MONGO_COLLECTION_PARRAFOS", "")
MONGODB_USER_ENCODED = urllib.parse.quote_plus(MONGODB_USER)
MONGODB_PASSWORD_ENCODED = urllib.parse.quote_plus(MONGODB_PASSWORD)
MONGODB_URI = f"mongodb://{MONGODB_USER_ENCODED}:{MONGODB_PASSWORD_ENCODED}@{MONGODB_HOST}:{MONGODB_PORT}/{MONGODB_PORT}?authSource={MONGODB_AUTH_SOURCE}"

LOG_DIR = "logs"
ARCHIVO_LOG="app.log"

MODELO=os.getenv("MODELO", "")

ZONA_HORARIA=os.getenv("ZONA_HORARIA", "")
puerto_txt=os.getenv("PUERTO_API_RESTO", "")
PUERTO=int(puerto_txt)

DIR_FAISS = os.path.join(BASE_DIR, "07_FAISS")
DIR_CHROMA = os.path.join(BASE_DIR, "07_CHROMADB")
CHROMA_COLECCION= os.getenv("CHROMA_COLECCION","") 
FAISS_ARCHIVO_INDICE=  os.getenv("FAISS_ARCHIVO_INDICE","")  



##############################################################################
# CONECCIÓN MONGO PARA leer metadatos
##############################################################################
client = pymongo.MongoClient(MONGODB_URI)
db = client[MONGO_DB]
coleccion_faiss = db[MONGO_COLLECTION_FAISS]
coleccion_parrafos = db[MONGO_COLLECTION_PARRAFOS]

##############################################################################
# CONFIGURACIÓN PARA LOGS
##############################################################################
# Definir la zona horaria de Guayaquil para log
zona_horaria_guayaquil = pytz.timezone(ZONA_HORARIA)

def guayaquil_time(*args):
    """Convierte la hora UTC a la zona horaria de Guayaquil."""
    utc_dt = datetime.utcnow().replace(tzinfo=pytz.utc)
    local_dt = utc_dt.astimezone(zona_horaria_guayaquil)
    return local_dt.timetuple()

# Crear carpeta logs si no existe
os.makedirs(LOG_DIR, exist_ok=True)
# Configurar el logger con zona horaria de Guayaquil
LOG_FILE = os.path.join(LOG_DIR, ARCHIVO_LOG)

logging.Formatter.converter = guayaquil_time  # Usar la conversión a zona horaria local
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),  # Guardar logs en archivo
        logging.StreamHandler()  # Mostrar logs en consola
    ]
)

logger = logging.getLogger(__name__)


##############################################################################
# FAST API INICIALIZACIÓN
##############################################################################
app = FastAPI(title="API de Búsqueda en FAISS y ChromaDB",
              description="API para realizar búsquedas en FAISS y ChromaDB usando embeddings.",
              version="1.0")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas las solicitudes (puedes cambiarlo a tu dominio específico)
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos HTTP (GET, POST, etc.)
    allow_headers=["*"],  # Permitir todos los encabezados
)

##############################################################################
# LECTURA DE INDICES
##############################################################################

device = "mps" if torch.backends.mps.is_available() else "cpu"
# Cargar modelo de embeddings

modelo_legal = SentenceTransformer(MODELO, device=device)
logging.info(f"✅ Modelo cargado en: {modelo_legal.device}")

# Cargar índice FAISS
INDEX_FILE = os.path.join(DIR_FAISS, FAISS_ARCHIVO_INDICE)
embedding_dim = 768

if os.path.exists(INDEX_FILE):
    index_faiss = faiss.read_index(INDEX_FILE)
    logging.info("✅ Índice FAISS cargado.")
else:
    logging.warning("⚠️ Índice FAISS vacío o no encontrado.")

# Cargar base de datos ChromaDB
chroma_client = chromadb.PersistentClient(path=DIR_CHROMA)
chroma_collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLECCION,
    metadata={"hnsw:space": "cosine"}  # Puedes cambiar "cosine" por "l2" o "ip"
    )
'''
	•	"l2" → Distancia Euclidiana (buena para datos densos)
	•	"ip" → Producto interno (útil si los embeddings están normalizados)
	•	"cosine" → Similitud coseno (recomendada para texto)
'''

logging.info("✅ Base de datos ChromaDB cargada.")
logging.info("##############################################################")
logging.info(f"INDICES: {BASE_DIR}")
logging.info("##############################################################")



##############################################################################
# FUNCIONES PARA EXPONER API REST FAISS
##############################################################################

@app.get("/buscar/faiss")################################################################################
def buscar_en_faiss(query: str = Query(..., description="Texto a buscar en FAISS"), top_k: int = 5):
    """
    API REST para realizar una búsqueda en FAISS y devolver los resultados más relevantes.
    """
    try:
        #vector_query = modelo_legal.encode([query], device=device)
        #distances, indices = index_faiss.search(np.array(vector_query).astype('float32'), top_k)
        
        vector_query = modelo_legal.encode([query], device=device)  #Generar el embedding de la consulta
        vector_query = np.array(vector_query).astype('float32')  #Convertir a float32 para FAISS (sin normalizar)
        #vector_query /= np.linalg.norm(vector_query)   #Normalizar el vector si trabajamos con similitud coseno
        distances, indices = index_faiss.search(vector_query, top_k)
        

        resultados = []
        for i in range(top_k):
            if indices[0][i] == -1:
                continue  # Ignorar resultados inválidos
            indice_faiss=int(indices[0][i])
            
            metadata = coleccion_faiss.find_one({"indice_faiss": indice_faiss}, {"_id": 0})  # Excluir _id de Mongo

            resultados.append({
                "indice": indice_faiss,
                "distancia": float(distances[0][i]),
                "metadata": metadata if metadata else {}
            })
        return {"query": query, "resultados": resultados}

    except Exception as e:
        logging.error(f"❌ Error en búsqueda FAISS: {e}\n{traceback.format_exc()}")
        return {"error": str(e)}


##############################################################################
# FUNCIONES PARA EXPONER API REST CHROMADB
##############################################################################

@app.get("/buscar/chromadb")################################################################################
def buscar_en_chromadb(query: str = Query(..., description="Texto a buscar en ChromaDB"), top_k: int = 5,filtro=None):
    """
    API REST para realizar una búsqueda en ChromaDB y devolver los resultados más relevantes.
    """
    try:
        #filtro={"articulo_numero": "173"}
        
        vector_query = modelo_legal.encode([query], device=device)[0].tolist()
        resultados = chroma_collection.query(
            query_embeddings=[vector_query],
            n_results=top_k,
            where=filtro  
        )

        documentos = []
        for i in range(len(resultados["ids"][0])):
            documentos.append({
                "id": resultados["ids"][0][i],
                "distancia": resultados["distances"][0][i],
                "metadata": resultados["metadatas"][0][i]
            })

        return {"query": query, "resultados": documentos}

    except Exception as e:
        logging.error(f"❌ Error en búsqueda ChromaDB: {e}\n{traceback.format_exc()}")
        return {"error": str(e)}



##############################################################################
# FUNCIONES PARA EXPONER METADATA
##############################################################################
@app.get("/documentos", summary="Obtener todos los documentos")
def obtener_documentos():
    """
    Devuelve todos los documentos almacenados en la colección `documentos_parrafos`.
    """
    documentos = []
    for doc in coleccion_parrafos.find({}, {"_id": 0, "texto": 0 }):  # Excluir `_id`
        documentos.append(doc)
    
    return {"documentos": documentos}



##############################################################################
# FUNCIONES PARA EXPONER LLMS
##############################################################################
@app.get("/buscar/llms")
def buscar_llms(mensaje: str = Query(..., description="Mensaje a repetir")):
    """
    API REST que devuelve el mismo mensaje enviado.
    """
    try:
        logging.info(f"📢 Eco recibido: {mensaje}")
        mensaje='Dijiste:'+mensaje
        return {"mensaje_recibido": mensaje, "respuesta": f"Eco: {mensaje}"}
    
    
    
    except Exception as e:
        logging.error(f"❌ Error en el endpoint de eco: {e}\n{traceback.format_exc()}")
        return {"error": str(e)}



##############################################################################
# INICIO PROGRAMA
##############################################################################
# Ejecución del servidor
if __name__ == "__main__":
    import uvicorn
    logging.info("🚀 Iniciando servidor FastAPI...")
    uvicorn.run(app, host="0.0.0.0", port=PUERTO)