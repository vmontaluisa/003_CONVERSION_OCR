import chromadb
import torch
import uvicorn
from fastapi import FastAPI
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

# Configuración del modelo
CHROMA_DB_PATH = "07_CHROMADB"
LLAMA_MODEL_PATH = "models/llama-2-7b.Q4_K_M.gguf"
EMBEDDING_MODEL = "hiiamsid/sentence_similarity_spanish_es"
from fastapi import FastAPI, Query
import requests
import chromadb
import spacy
from sentence_transformers import SentenceTransformer
import uvicorn

# Configuración de la API de Llama 2
LLAMA_API_URL = "http://localhost:7000/completion"

# Configuración de ChromaDB
CHROMA_DB_PATH = "07_CHROMADB"  # Ruta de ChromaDB
CHROMA_COLLECTION = "documentos_legales"

# Cargar modelo de embeddings
EMBEDDING_MODEL = "hiiamsid/sentence_similarity_spanish_es"
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Cargar modelo de procesamiento de texto en español
nlp = spacy.load("es_core_news_lg")

# Inicializar FastAPI
app = FastAPI(title="API RAG con Llama 2 y ChromaDB",
              description="API para responder preguntas usando Retrieval-Augmented Generation (RAG).",
              version="1.0")

# Inicializar ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
chroma_collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION)



def resumir_chunk(chunk):
    """Llama a Llama 2 para resumir un chunk grande de texto."""
    prompt_resumen = f"Resume este texto:\n{chunk}"
    return llama2_api_call(prompt_resumen)

def procesar_contexto_con_resumenes(contexto, pregunta):
    """Resume cada chunk del contexto y luego pregunta a Llama 2."""
    chunks = dividir_en_chunks(contexto, tamano_chunk=1500)  # Dividir en partes más grandes
    resúmenes = [resumir_chunk(chunk) for chunk in chunks]  # Obtener resúmenes
    contexto_resumido = "\n\n".join(resúmenes)  # Unir los resúmenes

    prompt_final = f"Contexto resumido:\n{contexto_resumido}\n\nPregunta: {pregunta}\n\nRespuesta:"
    return llama2_api_call(prompt_final)


def preprocesar_texto(texto):
    """
    Aplica lematización y eliminación de stopwords a una consulta en español.
    """
    doc = nlp(texto.lower())  # Convertir a minúsculas
    palabras_filtradas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(palabras_filtradas)

def search_chroma(query, top_k=5):
    """
    Realiza una búsqueda en ChromaDB y devuelve los documentos más relevantes.
    """
    query_preprocesada = preprocesar_texto(query)
    query_embedding = embedding_model.encode([query_preprocesada])[0].tolist()

    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    documentos = []
    for i in range(len(results["ids"][0])):
        documentos.append(results["metadatas"][0][i]["parrafo_texto_original"])

    return documentos

def generate_llama_response(contexto, pregunta):
    """
    Genera una respuesta usando la API REST de Llama 2.
    """
    prompt = f"""
    Basado en la siguiente información:
    {contexto}

    Responde la siguiente pregunta de manera clara y concisa:
    {pregunta}
    """

    response = requests.post(LLAMA_API_URL, json={"prompt": prompt, "n_predict": 200})

    if response.status_code == 200:
        return response.json().get("content", "No se pudo generar una respuesta.")
    else:
        return f"Error en Llama API: {response.text}"

def rag_pipeline(pregunta):
    """
    Pipeline RAG: Recupera documentos con ChromaDB y genera respuesta con Llama 2.
    """
    documentos_relevantes = search_chroma(pregunta, top_k=3)

    if not documentos_relevantes:
        return "No encontré información relevante en la base de datos."

    contexto = "\n\n".join(documentos_relevantes)

    respuesta = generate_llama_response(contexto, pregunta)

    return respuesta

# 📌 Endpoint para responder preguntas usando RAG
#@app.get("/rag", summary="Responder pregunta con RAG")
def responder_pregunta(pregunta: str = Query(..., description="Pregunta del usuario")):
    """
    API REST para realizar preguntas utilizando Retrieval-Augmented Generation (RAG).
    """
    respuesta = rag_pipeline(pregunta)
    return {"pregunta": pregunta, "respuesta": respuesta}

# 📌 Ejecutar API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)