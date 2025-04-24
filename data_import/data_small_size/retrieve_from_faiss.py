import json
import os
import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

class FaissRetriever:
    def __init__(self, index_path, anforande_ids_path, documents_path):
        self.index = faiss.read_index(index_path)
        self.anforande_ids = np.load(anforande_ids_path, allow_pickle=True)
        self.documents = np.load(documents_path, allow_pickle=True)

    def retrieve(self, query, top_k=10):
        query_embedding = np.array(embedding.embed_query(query), dtype="float32").reshape(1, -1)
        
        # Normalize the query for cosine similarity (since index vectors are normalized too)
        faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i in range(top_k):
            idx = indices[0][i]
            similarity = distances[0][i]
            anforande_id = self.anforande_ids[idx]
            doc = self.documents[idx]
            results.append((anforande_id, doc, similarity))
        return results



if __name__ == "__main__":
    # Initialize the FaissRetriever with the paths to your files
    retriever = FaissRetriever(
        index_path="/mnt/c/Users/User/thesis/data_import/data_small_size/data/index/faiss_index.bin",
        anforande_ids_path="/mnt/c/Users/User/thesis/data_import/data_small_size/data/index/anforande_ids.npy",
        documents_path="/mnt/c/Users/User/thesis/data_import/data_small_size/data/index/documents.npy"
    )
    
    # Query for retrieval
    query = "Vad säger Magdalena Andersson om kriget i Ukraina?"
    top_matches = retriever.retrieve(query)

    # Print the top matches
    print("Top matches:")
    for anforande_id, match, score in top_matches:
        print(f"Anförande ID: {anforande_id} | Similarity: {score:.4f}")
        print(f"Document: {match[:200]}...\n")
        print("""
                       __
                      / _)
             .----._/ /
            /          |
         __/  (  |  (  /
        /__.-'|_|--|__|
        """)
