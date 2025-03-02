pip freeze > requirements.txt

python -m spacy download es_core_news_sm


------------------
python3 -m venv venv
source venv/bin/activate

---------------------------
mongod --config /opt/homebrew/etc/mongod.conf
brew services stop mongodb/brew/mongodb-community



----------------------
import requests

query = "Asamblea Constituyente"

# Llamada a FAISS
response_faiss = requests.get(f"http://127.0.0.1:8000/buscar/faiss", params={"query": query, "top_k": 5})
print("FAISS:", response_faiss.json())

# Llamada a ChromaDB
response_chroma = requests.get(f"http://127.0.0.1:8000/buscar/chromadb", params={"query": query, "top_k": 5})
print("ChromaDB:", response_chroma.json())