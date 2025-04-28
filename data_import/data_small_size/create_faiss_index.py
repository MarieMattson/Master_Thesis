import os
import json
import time
from more_itertools import chunked
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
    try:
        date = item.get("dok_datum", "").strip()
        rubrik = item.get("avsnittsrubrik", "").strip()
        talare = item.get("talare", "").strip()
        text = item.get("anforandetext", "").strip()
    except AttributeError:
        print(f"⚠️ Missing attribute in item: {item}")
        continue

    full_text = f"{date}: {rubrik}. {talare}: {text}"
    if full_text.strip():
        texts.append(full_text)
        anforande_ids.append(item.get("anforande_id", "unknown_id"))

# Embedding with concurrency
print("🔁 Generating embeddings with concurrency...")
embedding_start = time.time()

batch_size = 16

def embed_batch(batch):
    return embedding_model.embed_documents(batch)

embeddings = []
for batch in tqdm(chunked(texts, batch_size), total=len(texts) // batch_size + 1):
    try:
        result = embed_batch(batch)
        embeddings.extend(result)
        time.sleep(1)  # avoid rate limits
    except Exception as e:
        print(f"❌ Error in batch: {e}")

embedding_duration = time.time() - embedding_start
print(f"⏱️ Embedding took {embedding_duration:.2f} seconds")

# Check for any issues before converting to NumPy
if not all(isinstance(vec, list) and all(isinstance(x, float) for x in vec) for vec in embeddings):
    raise ValueError("⚠️ Detected invalid embeddings. Some vectors are malformed or None.")

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
