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



# Cargar configuraciones desde .env
load_dotenv()



MONGO_DB = os.getenv("MONGO_DB", "documentos_legales")
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



client = pymongo.MongoClient(MONGODB_URI)
db = client[MONGO_DB]
coleccion_faiss = db[MONGO_COLLECTION_FAISS]


# Definir la zona horaria de Guayaquil
zona_horaria_guayaquil = pytz.timezone("America/Guayaquil")

def guayaquil_time(*args):
    """Convierte la hora UTC a la zona horaria de Guayaquil."""
    utc_dt = datetime.utcnow().replace(tzinfo=pytz.utc)
    local_dt = utc_dt.astimezone(zona_horaria_guayaquil)
    return local_dt.timetuple()

# Crear carpeta logs si no existe
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Configurar el logger con zona horaria de Guayaquil
LOG_FILE = os.path.join(LOG_DIR, "app.log")

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




# Cargar variables de entorno
load_dotenv()


# Iniciar FastAPI
app = FastAPI(title="API de Búsqueda en FAISS y ChromaDB",
              description="API para realizar búsquedas en FAISS y ChromaDB usando embeddings.",
              version="1.0")

# Verificar si MPS está disponible en Mac
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Cargar modelo de embeddings
modelo_legal = SentenceTransformer("PlanTL-GOB-ES/roberta-base-bne", device=device)
logging.info(f"✅ Modelo cargado en: {modelo_legal.device}")

# Configurar rutas de los índices
BASE_DIR = os.getenv("BASE_DIR", "../REPOSITORIO_DOCUMENTOS/preprocesamiento")
DIR_FAISS = os.path.join(BASE_DIR, "07_FAISS")
DIR_CHROMA = os.path.join(BASE_DIR, "07_CHROMADB")

# Cargar índice FAISS
INDEX_FILE = os.path.join(DIR_FAISS, "faiss_index.idx")
embedding_dim = 768

if os.path.exists(INDEX_FILE):
    index_faiss = faiss.read_index(INDEX_FILE)
    logging.info("✅ Índice FAISS cargado.")
else:
    index_faiss = faiss.IndexFlatL2(embedding_dim)  # Crear índice vacío si no existe
    logging.warning("⚠️ Índice FAISS vacío o no encontrado.")

# Cargar base de datos ChromaDB
chroma_client = chromadb.PersistentClient(path=DIR_CHROMA)
chroma_collection = chroma_client.get_or_create_collection(name="documentos_legales")
logging.info("✅ Base de datos ChromaDB cargada.")


@app.get("/buscar/faiss")
def buscar_en_faiss(query: str = Query(..., description="Texto a buscar en FAISS"), top_k: int = 5):
    """
    API REST para realizar una búsqueda en FAISS y devolver los resultados más relevantes.
    """
    try:
        vector_query = modelo_legal.encode([query], device=device)
        distances, indices = index_faiss.search(np.array(vector_query).astype('float32'), top_k)

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


@app.get("/buscar/chromadb")
def buscar_en_chromadb(query: str = Query(..., description="Texto a buscar en ChromaDB"), top_k: int = 5):
    """
    API REST para realizar una búsqueda en ChromaDB y devolver los resultados más relevantes.
    """
    try:
        vector_query = modelo_legal.encode([query], device=device)[0].tolist()
        resultados = chroma_collection.query(
            query_embeddings=[vector_query],
            n_results=top_k
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


# Ejecución del servidor
if __name__ == "__main__":
    import uvicorn
    logging.info("🚀 Iniciando servidor FastAPI...")
    uvicorn.run(app, host="0.0.0.0", port=8000)