import os
import json
import time
import numpy as np
import faiss
from langchain_openai import OpenAIEmbeddings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Start timing
full_start = time.time()

# Set up embedding model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Load dataset
with open("/mnt/c/Users/User/thesis/data_import/data_small_size/data/dataset_small.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Build combined texts and IDs
texts = []
anforande_ids = []

for item in data:
    rubrik = item.get("avsnittsrubrik", "").strip()
    talare = item.get("talare", "").strip()
    text = item.get("anforandetext", "").strip()
    
    full_text = f"{rubrik}. {talare}: {text}"
    if full_text.strip():
        texts.append(full_text)
        anforande_ids.append(item.get("anforande_id", "unknown_id"))

# Embedding with concurrency
print("🔁 Generating embeddings with concurrency...")
embedding_start = time.time()

def embed_text(text):
    return embedding_model.embed_query(text)

embeddings = [None] * len(texts)
with ThreadPoolExecutor(max_workers=8) as executor:
    future_to_index = {executor.submit(embed_text, text): i for i, text in enumerate(texts)}
    for future in tqdm(as_completed(future_to_index), total=len(future_to_index)):
        i = future_to_index[future]
        try:
            embeddings[i] = future.result()
        except Exception as e:
            print(f"❌ Error embedding text at index {i}: {e}")

embedding_duration = time.time() - embedding_start
print(f"⏱️ Embedding took {embedding_duration:.2f} seconds")

# Convert to NumPy
embeddings_np = np.array(embeddings, dtype="float32")

# Normalize the embeddings to unit vectors (important for cosine similarity)
faiss.normalize_L2(embeddings_np)

# Create FAISS index with dot product (cosine similarity)
index = faiss.IndexFlatIP(embeddings_np.shape[1])
index.add(embeddings_np)

# Save everything
os.makedirs("data/index", exist_ok=True)
faiss.write_index(index, "data/index/faiss_index.bin")
np.save("data/index/anforande_ids.npy", np.array(anforande_ids))
np.save("data/index/documents.npy", np.array(texts))

# Done
full_duration = time.time() - full_start
print("✅ FAISS index created and saved.")
print(f"🕒 Total time: {full_duration:.2f} seconds")
