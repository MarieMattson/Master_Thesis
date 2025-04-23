import json
import os
from IPython import embed
import faiss
from langchain_neo4j import Neo4jGraph
from neo4j import GraphDatabase
import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD =  os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
graph = Neo4jGraph()

class CosineRAG:
    def __init__(self):
        self.data_path = "/mnt/c/Users/User/thesis/data_import/data_small_size/dataset_small.json"
        self.llm = ChatOpenAI(model="gpt-4o")
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        self.embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        self.documents = []
        self.anforande_ids = []
        self.embeddings = []
        self.index = None

        if os.path.exists("/mnt/c/Users/User/thesis/data_import/data_small_size/index/faiss_index.bin"):
            print("Loading existing FAISS index...")
            self.index = faiss.read_index("//mnt/c/Users/User/thesis/data_import/data_small_size/index/faiss_index.bin")
            self.anforande_ids = np.load("/mnt/c/Users/User/thesis/data_import/data_small_size/index/anforande_ids.npy", allow_pickle=True)
            self.documents = np.load("/mnt/c/Users/User/thesis/data_import/data_small_size/index/documents.npy", allow_pickle=True)
        else:
            # Load data and process if no index exists
            print("Creating new FAISS index...")
            self.data = self.load_data()
            self.process_documents()

    def process_documents(self):
        self.speaker_cache = {}
        self.title_cache = {}

        for item in self.data:
            title = item.get("dok_titel", "")
            text = item.get("anforandetext", "")
            speaker = item.get("talare", "")
            anforande_id = item.get("anforande_id", "")
            
            if text:
                full_text = f"{title}: {speaker}: {text}"
                self.documents.append(full_text)
                self.anforande_ids.append(anforande_id)
                # Get speaker embedding (with cache)
                if speaker in self.speaker_cache:
                    embedded_speaker = self.speaker_cache[speaker]
                else:
                    embedded_speaker = np.array([self.embedding.embed_query(speaker)]).astype("float32")
                    self.speaker_cache[speaker] = embedded_speaker

                # Get title embedding (with cache)
                if title in self.title_cache:
                    embedded_title = self.title_cache[title]
                else:
                    embedded_title = np.array([self.embedding.embed_query(title)]).astype("float32")
                    self.title_cache[title] = embedded_title
                text_embedding = self.get_embedding(anforande_id)
                combined_embedding = np.concatenate([embedded_title[0], embedded_speaker[0], text_embedding])
                self.embeddings.append(combined_embedding)                
                print(f"Embedding created for anförande {anforande_id}")

        embeddings_np = np.array(self.embeddings).astype("float32")
        dimension = embeddings_np.shape[1]

        self.index = faiss.IndexFlatIP(dimension)  # Using Inner Product for cosine similarity
        self.index.add(embeddings_np)

        index_dir = "/mnt/c/Users/User/thesis/data_import/data_small_size/index"
        os.makedirs(index_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(index_dir, "faiss_index.bin"))
        np.save(os.path.join(index_dir, "anforande_ids.npy"), np.array(self.anforande_ids))
        np.save(os.path.join(index_dir, "documents.npy"), np.array(self.documents))

    def get_embedding(self, anforande_id):
        cypher_query = "MATCH (c:Chunk {anforande_id: $anforande_id}) RETURN c.embedding AS embedding"
        with self.driver.session() as session:
                result = session.run(cypher_query, anforande_id=anforande_id)
                record = result.single()

                if record and record["embedding"]:
                    return np.array(record["embedding"], dtype=np.float32)
                
    def load_data(self):
        """Load JSON data from the file."""
        with open(self.data_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    
    def close(self):
        if self.driver:
            self.driver.close()


if __name__ == "__main__":
    rag = CosineRAG()
    rag.close() 