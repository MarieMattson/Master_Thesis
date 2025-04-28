import json
import os
import faiss
import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage
from dotenv import load_dotenv
load_dotenv()
EDENAI_API_KEY = os.getenv("EDENAI_API_KEY")
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

class FaissRetriever:
    def __init__(self, index_path, anforande_ids_path, documents_path):
        self.index = faiss.read_index(index_path)
        self.anforande_ids = np.load(anforande_ids_path, allow_pickle=True)
        self.documents = np.load(documents_path, allow_pickle=True)
        
        self.url = "https://api.edenai.run/v2/text/chat"
        self.headers = {"Authorization": f"Bearer {EDENAI_API_KEY}",
                        "accept": "application/json",
                        "content-type": "application/json"}
        
        self.payload = {
                        "providers": "meta/llama3-1-405b-instruct-v1:0",  # "openai/gpt-4o",   #"deepseek/DeepSeek-V3",
                        "response_as_dict": True,
                        "attributes_as_list": False,
                        "show_base_64": True,
                        "show_original_response": False,
                        "temperature": 0,
                        "max_tokens": 4096,
                        "tool_choice": "auto",
                        "previous_history": [
                            {'role': 'user', 'message': f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:\n{answer}"}, 
                            {'role': 'user', 'message': global_message}
                        ]
                    }

        self.llm = ChatOpenAI(model="gpt-4o")

    def retrieve(self, query, top_k=6):
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
    
    def generate_response(self, ranked_speeches:list, user_query:str):
        print(ranked_speeches)
        
        prompt = f"User query: {user_query}\n\nBased on the ranked debate chunks, generate a relevant response to the user query using the following information You must respond in Swedish:\n\n"
        for i, (anforande_id, anforande, similarity) in enumerate(ranked_speeches, start=1):
            prompt += f"\n🔹 **Result {i}:**\n"
            prompt += f"🗣 **Anförande Text:** {anforande}...\n"
        prompt += "\nNow, generate a response based on the context provided above. Make sure to answer the user query in a natural and coherent way based on the information from the debate. You must respond in Swedish"
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        return response.content

if __name__ == "__main__":
    # Initialize the FaissRetriever with the paths to your files
    retriever = FaissRetriever(
        index_path="/mnt/c/Users/User/thesis/data_import/data_small_size/data/index/faiss_index.bin",
        anforande_ids_path="/mnt/c/Users/User/thesis/data_import/data_small_size/data/index/anforande_ids.npy",
        documents_path="/mnt/c/Users/User/thesis/data_import/data_small_size/data/index/documents.npy"
    )
    
    # Query for retrieval
    query = "Vad säger Magdalena Andersson om kriget i Ukraina?"
    ranked_speeches = retriever.retrieve(query)
    
    response = retriever.generate_response(ranked_speeches, query)

    # Print the generated response
    print("\nGenerated Response:\n")
    print(response)
    
    # Print the top matches (optional)
    print("\nTop matches:")
    for anforande_id, match, score in ranked_speeches:
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
