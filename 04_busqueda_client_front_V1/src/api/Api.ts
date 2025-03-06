import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL  || "http://127.0.0.1:8000" ;
const API_BASE_URL_RAG = import.meta.env.VITE_API_BASE_URL  || "http://127.0.0.1:8001" ;
//const API_BASE_URL = import.meta.env.VITE_API_BASE_URL  || "https://hr0913c6-8000.use2.devtunnels.ms/" ;



console.log("API_BASE_URL:", API_BASE_URL);

// Obtener lista de documentos
export const getDocuments = async () => {
  const response = await axios.get(`${API_BASE_URL}/documentos`);
  return response.data;
};

/// Buscar en ChromaDB con cantidad de resultados (topK)
export const searchChromaDB = async (query: string, topK: number = 5) => {
  const response = await axios.get(`${API_BASE_URL}/buscar/chromadb`, {
    params: { query, top_k: topK },
  });
  return response.data;
};

// Buscar en FAISS con cantidad de resultados (topK)
export const searchFAISS = async (query: string, topK: number = 5) => {
  const response = await axios.get(`${API_BASE_URL}/buscar/faiss`, {
    params: { query, top_k: topK },
  });
  return response.data;
};

// Buscar en el chatbot
export const searchLLMS = async (mensaje: string) => {
  const response = await axios.get(`${API_BASE_URL_RAG}/rag`, {
    params: { mensaje },
  });
  return response.data;
};