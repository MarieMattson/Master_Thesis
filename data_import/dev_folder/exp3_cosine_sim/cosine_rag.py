import enum
import json
import os
import faiss
import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage, SystemMessage
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class CosineRAG:
    def __init__(self, data_path=None, api_key=None, chunk_size=500, chunk_overlap=50):
        self.data_path = "/mnt/c/Users/User/thesis/data_import/filtered_riksdag_exp1.json"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.embedding_model = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm = ChatOpenAI(model="gpt-4")
        
        self.documents = []
        self.anforande_ids = []
        self.embeddings = []
        self.index = None

        self.url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }

        if os.path.exists("/mnt/c/Users/User/thesis/data_import/exp3_cosine_sim/index/faiss_index.bin"):
            print("Loading existing FAISS index...")
            self.index = faiss.read_index("/mnt/c/Users/User/thesis/data_import/exp3_cosine_sim/index/faiss_index.bin")
            self.anforande_ids = np.load("/mnt/c/Users/User/thesis/data_import/exp3_cosine_sim/index/anforande_ids.npy", allow_pickle=True)
            self.documents = np.load("/mnt/c/Users/User/thesis/data_import/exp3_cosine_sim/index/documents.npy", allow_pickle=True)
        else:
            # Load data and process if no index exists
            print("Creating new FAISS index...")
            self.data = self.load_data()
            self.process_documents()


    def load_data(self):
        """Load JSON data from the file."""
        with open(self.data_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data


    def process_documents(self):
        # Convert each data item to a LangChain Document
        documents = [
            Document(page_content=item["anforandetext"], metadata={"anforande_id": item["anforande_id"]})
            for item in self.data if "anforandetext" in item
        ]

        text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = text_splitter.split_documents(documents)
        print(f"Number of chunks: {len(chunks)}")
        for chunk in chunks[:5]:
            print(f"Chunk content: {chunk.page_content[:200]}...")  # Print a sample from each chunk

        for item in self.data:
            title = item.get("dok_titel", "")
            text = item.get("anforandetext", "")
            anforande_id = item.get("anforande_id", "")
            
            if text:
                full_text = f"{title}: {text}"
                self.documents.append(full_text)
                self.anforande_ids.append(anforande_id)
                self.embeddings.append(self.embedding_model.embed_query(full_text))

        self.embeddings = np.array(self.embeddings).astype("float32")

        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Using Inner Product for cosine similarity
        self.index.add(self.embeddings)

        faiss.write_index(self.index, "faiss_index.bin")
        np.save("anforande_ids.npy", np.array(self.anforande_ids))
        np.save("documents.npy", np.array(self.documents))


    def retrieve(self, query, top_k=6):
        """Retrieve the top k most similar documents for a given query."""
        query_embedding = np.array([self.embedding_model.embed_query(query)]).astype("float32")
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i in range(top_k):
            idx = indices[0][i]
            similarity = distances[0][i]
            anforande_id = self.anforande_ids[idx]
            doc = self.documents[idx]
            results.append((anforande_id, doc, similarity))
        return results
    
    def generate_response(self, top_matches, user_query=str):
        prompt = f"User query: {user_query}\n\nBased on the ranked debate chunks, generate a relevant response to the user query using the following information You must respond in Swedish:\n\n"
        for i, (anforande_id, anforande_text, score) in enumerate(top_matches):
            prompt += f"\n🔹 **Result {i}:**\n"
            prompt += f"🗣 **Anförande Text:** {anforande_text}...\n"
            prompt += f"⭐ **Similarity Score:** {score:.4f}\n"
        prompt += "\nNow, generate a response based on the context provided above. Make sure to answer the user query in a natural and coherent way based on the information from the debate. You must respond in Swedish"
        response = self.llm.invoke([HumanMessage(content=prompt)])

        return response.content

if __name__ == "__main__":
    data_path = "/mnt/c/Users/User/thesis/data_import/filtered_riksdag_exp1.json"
    riksdag = CosineRAG(data_path)
    
    query = "Vad tycker Per Bolund om vindkraft?"
    top_matches = riksdag.retrieve(query)

    print("Top matches:")
    for anforande_id, match, score in top_matches:
        print(f"Anförande ID: {anforande_id} | Similarity: {score:.4f}")
        print(f"Document: {match[:200]}...\n")

    response = riksdag.generate_response(top_matches, query)
    print("\n🧠 Genererat Svar:\n")
    print(response)