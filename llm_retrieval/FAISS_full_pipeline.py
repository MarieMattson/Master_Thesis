import json
import os
import faiss
import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

class FaissRetriever:
    def __init__(self, index_path, anforande_ids_path, documents_chunks):
        self.index = faiss.read_index(index_path)
        self.anforande_ids = np.load(anforande_ids_path, allow_pickle=True)
        #self.documents = documents
        self.documents_chunks = documents_chunks 
        self.chunk_sizes = [len(chunk) for chunk in documents_chunks]
        self.chunk_offsets = np.cumsum([0] + self.chunk_sizes)

        self.url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {OPENAI_API_KEY}"
                        }
        self.llm = ChatOpenAI(model="gpt-4o")


    def _get_doc_by_global_idx(self, global_idx):
        # Find which chunk the index belongs to
        chunk_id = np.searchsorted(self.chunk_offsets, global_idx, side='right') - 1
        local_idx = global_idx - self.chunk_offsets[chunk_id]
        return self.documents_chunks[chunk_id][local_idx]

    def retrieve(self, query, top_k=6):
        query_embedding = np.array(embedding.embed_query(query), dtype="float32").reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i in range(top_k):
            idx = indices[0][i]
            similarity = distances[0][i]
            anforande_id = self.anforande_ids[idx]
            doc = self._get_doc_by_global_idx(idx)
            results.append((anforande_id, doc, similarity))
        return results


    def generate_response(self, ranked_speeches:list, user_query:str):
        print(ranked_speeches)
        
        prompt = f"User query: {user_query}\n\nBased on the ranked debate chunks, generate a relevant response to the user query using the following information You must respond in Swedish:\n\n"
        for i, (anforande_id, anforande, similarity) in enumerate(ranked_speeches, start=1):
            prompt += f"\n🔹 **Result {i}:**\n"
            prompt += f"🗣 **Anförande Text:** {anforande}...\n"
        prompt += "\nNow, generate a response based on the context provided above. Make sure to answer the user query in a natural and coherent way based on the information from the debate. You must respond in Swedish"
        prompt += f"If no relevant information is found, respond with a fallback message.\n"
        prompt += f"Specify what is being asked in the question:\n\n"
        prompt += f"\"{user_query}\"\n\n"
        prompt += f"Respond: 'Jag hittar ingen information om ... i min data'."
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
