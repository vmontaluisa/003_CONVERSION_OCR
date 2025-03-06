import os
import requests
import chromadb
import spacy
import torch
import uvicorn
from fastapi import FastAPI, Query
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI  # Simula el API REST de Llama 2
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import urllib.parse
import json
from openai import OpenAI  # API de OpenAI
import traceback
from fastapi.middleware.cors import CORSMiddleware



# 🔥 Cargar variables de entorno
load_dotenv()


##############################################################################
#
# REPOSITORIO DE PROCESAMIENTO DE DOCUMENTOS
#  ../REPOSITORIO_DOCUMENTOS/preprocesamiento
#https://github.com/vmontaluisa/003_CONVERSION_OCR_preprocesamiento
BASE_DIR = os.getenv("BASE_DIR", "")
##############################################################################


DIR_CHROMA = os.path.join(BASE_DIR, os.getenv("DIR_CHROMA", "07_CHROMADB"))

##############################################################################
# CONFIGURACIÓN DE MONGO DB PARA ALMACENAMIENTO
##############################################################################
MONGO_DB = os.getenv("MONGO_DB", "")
MONGODB_HOST=os.getenv("MONGODB_HOST","")
MONGODB_PORT=os.getenv("MONGODB_PORT","")
MONGODB_AUTH_SOURCE=os.getenv("MONGODB_AUTH_SOURCE","")
MONGODB_USER=os.getenv("MONGODB_USER","")
MONGODB_PASSWORD=os.getenv("MONGODB_PASSWORD","")
MONGO_COLLECTION_CHROMA = os.getenv("MONGO_COLLECTION_CHROMA", "")
MONGO_COLLECTION_FAISS = os.getenv("MONGO_COLLECTION_FAISS", "")
MONGO_COLLECTION_PARRAFOS = os.getenv("MONGO_COLLECTION_PARRAFOS", "")

# Codificación de usuario y contraseña para conexión segura
MONGODB_USER_ENCODED = urllib.parse.quote_plus(MONGODB_USER)
MONGODB_PASSWORD_ENCODED = urllib.parse.quote_plus(MONGODB_PASSWORD)
MONGODB_URI = f"mongodb://{MONGODB_USER_ENCODED}:{MONGODB_PASSWORD_ENCODED}@{MONGODB_HOST}:{MONGODB_PORT}/{MONGODB_PORT}?authSource={MONGODB_AUTH_SOURCE}"


# Configuración de Llama 2 API REST
LLAMA_API_URL = "http://127.0.0.1:7000/completion"



#  Configuración de ChromaDB
CHROMA_COLECCION= os.getenv("CHROMA_COLECCION","") 
CHROMA_DB_PATH=f"{DIR_CHROMA}"  # Ruta donde se guardará la base de datos

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", "")
OPENAI_MODELO=os.getenv("OPENAI_MODELO", "")
TOTAL_RESULTADOS=15

# Cargar embeddings desde HuggingFace para LangChain
EMBEDDING_MODEL = "hiiamsid/sentence_similarity_spanish_es"
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


device = "mps" if torch.backends.mps.is_available() else "cpu"
# Cargar modelo de embeddings

modelo_legal = SentenceTransformer(EMBEDDING_MODEL, device=device)



#  Modelo de procesamiento de texto en español con spaCy
nlp = spacy.load("es_core_news_lg")


##############################################################################
# FAST API INICIALIZACIÓN
##############################################################################
app = FastAPI(title="API RAG con Llama 2 y ChromaDB",
              description="API para responder preguntas usando Retrieval-Augmented Generation (RAG).",
              version="1.0")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas las solicitudes (puedes cambiarlo a tu dominio específico)
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos HTTP (GET, POST, etc.)
    allow_headers=["*"],  # Permitir todos los encabezados
)

'''
#  Conectar ChromaDB con LangChain
chroma_db = Chroma(
    embedding_function=embedding_model,
    persist_directory=CHROMA_DB_PATH
)
'''



chroma_client = chromadb.PersistentClient(path=DIR_CHROMA)
chroma_collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLECCION,
    metadata={"hnsw:space": "cosine"}  # Puedes cambiar "cosine" por "l2" o "ip"
    )


# Simulación del Llama 2 API REST con LangChain
#llm = OpenAI(openai_api_base=LLAMA_API_URL, temperature=0)

# Función para preprocesar texto con spaCy (Lematización + Stopwords)
def preprocesar_texto(texto):
    """
    Aplica lematización y eliminación de stopwords a una consulta en español.
    """
    doc = nlp(texto.lower())  # Convertir a minúsculas
    palabras_filtradas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(palabras_filtradas)

##############################################################################
#  Función para buscar en ChromaDB 
def search_chroma_langchain(query, top_k=5):
    """
    Realiza una búsqueda en ChromaDB usando LangChain.
    """
    query_preprocesada = preprocesar_texto(query)
#    documentos = chroma_db.similarity_search(query_preprocesada, k=top_k)
    
    
    vector_query = modelo_legal.encode([query_preprocesada], device=device)[0].tolist()
    resultados = chroma_collection.query(
            query_embeddings=[vector_query],
            n_results=top_k,
        )

    documentos = []
    for i in range(len(resultados["ids"][0])):
            documentos.append({
                "id": resultados["ids"][0][i],
                "distancia": resultados["distances"][0][i],
                "metadata": resultados["metadatas"][0][i]
            })
    
    id_parrafos=[]
    contenidos=[]
    for doc_texto in documentos:        
        texto=doc_texto['metadata']['parrafo_texto_indexado']
        parrafo_unid=doc_texto['metadata']['parrafo_unid']
        
        contenidos.append(texto)
        id_parrafos.append(parrafo_unid)
    
    
    return contenidos,id_parrafos # Devuelve textos relevantes



##############################################################################

# 📝Función para generar respuesta con Llama 2 API REST
def generate_llama_response(contexto, pregunta):
    """
    Genera una respuesta usando Llama 2 API REST.
    """
    prompt = f"""
    Basado en la siguiente información recuperada:
    {contexto}

    Responde la siguiente pregunta de manera clara y concisa en un solo parrafo :
    {pregunta}
    """

    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(
        LLAMA_API_URL,
        json={"prompt": prompt, "n_predict": 200},
        headers=headers
        )

    if response.status_code == 200:
        return response.json().get("content", "No se pudo generar una respuesta.")
    else:
        return f"Error en Llama API: {response.text}"

##############################################################################


def generate_openai_response(contexto, pregunta):
    try:
            client = OpenAI(api_key=OPENAI_API_KEY)
                      
            prompt = f"""Basado en la siguiente información recuperada:
                            {contexto}
                        Responde la siguiente pregunta de manera clara y concisa en un solo parrafo :
                        {pregunta}
                        """
            print('Consultando openia inicio')
            response = client.chat.completions.create(
                model=OPENAI_MODELO,
                messages=[{"role": "system", "content": "Eres un asistente que extrae información clave exclusivamente del texto proporcionado. No hagas suposiciones."},
                        {"role": "user", "content": prompt}],
                temperature=0.0
            )            
            print('Consultando openia fin')
            respuesta =response.choices[0].message.content.strip()                    
            respuesta_transformada = {}
            return respuesta
    except Exception as e:
            print(f"Error al procesar metadata con OpenAI en : {e}\n{traceback.format_exc()}")
            return {}




##############################################################################

def rag_pipeline(pregunta: str):
    """
    Pipeline RAG sin RunnableParallel.
    - Busca en ChromaDB los documentos más relevantes.
    - Llama a Llama 2 para generar la respuesta final.
    """
    # 🔍 Buscar contexto en ChromaDB
    documentos_relevantes, ids_parrafos= search_chroma_langchain(pregunta, top_k=TOTAL_RESULTADOS)
    
    if not documentos_relevantes:
        return {"pregunta": pregunta, "respuesta": "No encontré información relevante en la base de datos."}
    
    contexto = "\n\n".join(documentos_relevantes)
    #respuesta = generate_llama_response(contexto, pregunta)
    respuesta =generate_openai_response(contexto, pregunta)

    return {"pregunta": pregunta, "respuesta": respuesta}


##############################################################################

# 📌 Endpoint para responder preguntas usando RAG
@app.get("/rag", summary="Responder pregunta con RAG")
def responder_pregunta(mensaje: str = Query(..., description="Pregunta del usuario")):
     return rag_pipeline(mensaje)


# 📌 Ejecutar API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)