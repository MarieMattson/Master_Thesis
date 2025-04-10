import json
import os
import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings
from data_import.embed import OPENAI_API_KEY
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


def retrieve(query, top_k=3):
    # Generate the query embedding
    query_embedding = np.array([embedding.embed_query(query)]).astype("float32")
    
    # Load the index and metadata
    index = faiss.read_index("faiss_index.bin")
    anforande_ids = np.load("anforande_ids.npy", allow_pickle=True)
    documents = np.load("documents.npy", allow_pickle=True)

    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for i in range(top_k):
        idx = indices[0][i]
        similarity = distances[0][i]
        anforande_id = anforande_ids[idx]
        doc = documents[idx]
        results.append((anforande_id, doc, similarity))
    return results

query = "Vad tycker Per Bolund om vindkraft?"
top_matches = retrieve(query)

print("Top matches:")
for anforande_id, match, score in top_matches:
    print(f"Anförande ID: {anforande_id} | Similarity: {score:.4f}")
    print(f"Document: {match[:200]}...\n")
