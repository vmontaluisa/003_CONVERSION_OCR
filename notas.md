pip freeze > requirements.txt

python -m spacy download es_core_news_sm
python -m spacy download es_core_news_md
python -m spacy download es_core_news_lg


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



==============================================================


# Instalar dependencias
brew install cmake make clang coreutils

brew install cmake make clang coreutils

brew install --cask clay
brew install cmake


brew install llvm
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"
export CC=/opt/homebrew/opt/llvm/bin/clang
export CXX=/opt/homebrew/opt/llvm/bin/clang++
export CMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang
export CMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++


export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"
export CC=/opt/homebrew/opt/llvm/bin/clang
export CXX=/opt/homebrew/opt/llvm/bin/clang++


===================


git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp


rm -rf build
mkdir build
cd build


cmake .. -DGGML_METAL=ON
cmake --build . --config Release


mkdir -p models
cd ~/llama.cpp/models
cd models
wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf
cd ..

./build/bin/llama-server -m models/llama-2-7b.Q4_K_M.gguf   --port 7000 --n-predict 512
./build/bin/llama-server -m models/llama-2-7b.Q4_K_M.gguf --threads 10  --port 7000 --n-predict 512

./build/bin/llama-server -m models/llama-2-7b.Q4_K_M.gguf --threads 8 --n-gpu-layers 20   --port 7000 --n-predict 512
./build/bin/llama-server -m models/llama-2-7b.Q4_K_M.gguf --threads 8 --n-gpu-layers 20   --port 7000 --n-predict 256


curl -X POST http://localhost:7000/completion \
     -H "Content-Type: application/json" \
     -d '{
          "prompt": "¿Qué es la inteligencia artificial?",
          "n_predict": 50,
          "temperature": 0.7,
          "top_k": 50,
          "top_p": 0.9,
          "repeat_penalty": 1.2
         }'


	•	n_predict: Reduce la cantidad de tokens generados (50 en lugar de 100+).
	•	top_k: Limita el número de palabras candidatas, acelerando la inferencia.
	•	top_p: Reduce la probabilidad de palabras irrelevantes.
	•	temperature: Mantén un valor bajo para generar texto más rápido y determinista.
	•	repeat_penalty: Evita repeticiones innecesarias, optimizando el uso de tokens.
