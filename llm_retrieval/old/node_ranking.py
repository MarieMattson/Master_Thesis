import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger
from neo4j import GraphDatabase
import numpy as np
import sklearn.metrics.pairwise
from llm_retrieval.old.node_retrieval import retrieve_node


load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
graph.refresh_schema()
enhanced_graph = Neo4jGraph(enhanced_schema=True)
llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

def rank_nodes_by_similarity(query_text: str, retrieved_nodes: list[dict], top_k=6) -> list[dict]:
    """Rank retrieved nodes based on cosine similarity with the user query."""
    
    query_embedding = np.array(embedding.embed_query(query_text)).reshape(1, -1)

    ranked_nodes = []
    for node in retrieved_nodes:
        chunk_embedding = node.get('c.embedding', None)
        if chunk_embedding:
            chunk_embedding = np.array(chunk_embedding).reshape(1,-1)
            similarity = sklearn.metrics.pairwise.cosine_similarity(query_embedding, chunk_embedding)
            ranked_nodes.append((node, similarity))

    ranked_nodes = sorted(ranked_nodes, key=lambda x: x[1], reverse=True)

    return [{"node": item[0], "score": item[1]} for item in ranked_nodes[:top_k]]

def print_ranked_nodes(ranked_nodes):
    """Print ranked nodes in a readable format without embeddings."""
    for i, item in enumerate(ranked_nodes, start=1):
        node = item["node"]
        score = item["score"]

        print(f"🔹 **Result {i}**")
        print(f"📌 **Chunk ID:** {node['c.chunk_id']}")
        print(f"🗣 **Anförande Text:** {node['a.anforande_text'][:300]}...")
        print(f"📜 **Chunk Text:** {node['c.text']}...")
        print(f"⭐ **Similarity Score:** {score[0][0]:.4f}")  
        print("-" * 80) 

if __name__ == "__main__":
    user_query = """
                Hur argumenterar JESSICA POLFJÄRD för sänkt restaurangmoms och fler jobb 
                i debatten från Protokoll H00998? Du MÅSTE returnera a.anforande_text, c.text, c.chunk_id och c.embedding!
                Generera en Cypher query som begränsar till den specifika debatten 
                och den aktuella talarens anförande, men undvik för många filter.
                """ 
    cypher_query ="""
                MATCH (t:Talare {name: "JESSICA POLFJÄRD"}) 
                MATCH (t)-[:HALLER]->(a:Anforande) 
                MATCH (a)-[:HAS_CHUNK]->(c:Chunk)
                MATCH (t)-[:DELTAR_I]->(d:Debatt)-[:DOCUMENTED_IN]->(p:Protokoll {dok_id: "H00998"})
                RETURN a.anforande_text, c.text, c.chunk_id, c.embedding        
                """
    retrieved_nodes = retrieve_node(cypher_query)

    ranked_nodes = rank_nodes_by_similarity(query_text= user_query, retrieved_nodes=retrieved_nodes)
    print(ranked_nodes)
    print_ranked_nodes(ranked_nodes=ranked_nodes)