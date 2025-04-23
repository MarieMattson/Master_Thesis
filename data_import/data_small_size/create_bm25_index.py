from rank_bm25 import BM25Okapi
import json
import os

file_path = "/mnt/c/Users/User/thesis/data_import/data_small_size/dataset_small.json" 

class BM25Index:
    def __init__(self, documents):
        self.documents = documents
        self.tokenized_corpus = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def query(self, query, top_n=5):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        return [(self.documents[i], scores[i]) for i in ranked_indices]

def load_documents(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    documents_path = "/mnt/c/Users/User/thesis/data_import/data_small_size/dataset_small.json"
    documents = load_documents(documents_path)

    bm25_index = BM25Index(documents)

    # Example query
    query = "Vad säger Lars Adaktusson om gängvåldet?"
    results = bm25_index.query(query)

    print("Top results:")
    for doc, score in results:
        print(f"Score: {score:.4f}, Document: {doc}")

if __name__ == "__main__":
    main()