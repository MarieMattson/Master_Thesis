import json
import os
import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


def retrieve(query, top_k=10):
    # Generate the query embedding
    query_embedding = np.array(embedding.embed_query(query), dtype="float32").reshape(1, -1)
    
    # Load the index and metadata
    index = faiss.read_index("/mnt/c/Users/User/thesis/data_import/data_small_size/data/index/faiss_index.bin")
    anforande_ids = np.load("/mnt/c/Users/User/thesis/data_import/data_small_size/data/index/anforande_ids.npy", allow_pickle=True)
    #documents = np.load("/mnt/c/Users/User/thesis/data_import/data_small_size/index/documents.npy", allow_pickle=True)
    print("Query embedding shape:", query_embedding.shape)
    print("Index dimension:", index.d)

    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for i in range(top_k):
        idx = indices[0][i]
        similarity = distances[0][i]
        anforande_id = anforande_ids[idx]
        #doc = documents[idx]
        results.append((anforande_id, similarity))
    return results

if __name__ == "__main__":
    query = "Vad tycker Per Bolund om vindkraft?"
    top_matches = retrieve(query)

    print("Top matches:")
    for anforande_id, match, score in top_matches:
        print(f"Anförande ID: {anforande_id} | Similarity: {score:.4f}")
        print(f"Document: {match[:200]}...\n")
        print("""
                       __
                      / _)
             _.----._/ /
            /          |
         __/  (  |  (  /
        /__.-'|_|--|__|
        """)