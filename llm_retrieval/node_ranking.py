import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger
from neo4j import GraphDatabase
import numpy as np
import sklearn.metrics.pairwise 


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

# Define system message for GraphCypherQAChain
#system_message = SystemMessage(
#    content="""
#            Your job is to rank nodes that are retrieved from a Neo4j database.
#            The data is transcribed debates from the Swedish parliament (Riksdagen).
#            You will be provided with a list of nodes with the following structure:
#            {'a.anforande_text': '\r\nHerr talman! Jag tänkte börja med att berätta hur...', 
#            'c.chunk_id': 'f28fefb8-ffeb-4ed9-a9da-4f0e7d356784', 
#            'c.embedding': [-0.008256875909864902, -0.01081767026335001, -0.019462913274765015....]}
#            Your job is to respond to user queries about Swedish parliamentary debates.
#            The data is in Swedish, and queries should not be overly specific.
#            Ensure the responses are relevant based on both structured Cypher queries and vector search results.
#            """
#)


def retrieve_node(query:str)->list[str]:
    logger.info("Connecting to Neo4j at {} as {}",NEO4J_URI,NEO4J_USERNAME)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
                result = session.run(query)
                nodes = [record.data() for record in result]
                return nodes
    finally:
        driver.close()


def rank_nodes_by_similarity(query_text: str, retrieved_nodes: list[dict], top_k=6) -> list[dict]:
    """Rank retrieved nodes based on cosine similarity with the user query."""
    
    # Generate embedding for the query
    query_embedding = np.array(embedding.embed_query(query_text)).reshape(1, -1)

    ranked_nodes = []
    for node in retrieved_nodes:
        chunk_embedding = node.get('c.embedding', None)
        if chunk_embedding:
            chunk_embedding = np.array(chunk_embedding).reshape(1,-1)
            similarity = sklearn.metrics.pairwise.cosine_similarity(query_embedding, chunk_embedding)
            ranked_nodes.append((node, similarity))

    # Sort nodes by similarity score in descending order
    ranked_nodes = sorted(ranked_nodes, key=lambda x: x[1], reverse=True)

    # Return top-k results
    return [{"node": item[0], "score": item[1]} for item in ranked_nodes[:top_k]]

def print_ranked_nodes(ranked_nodes):
    """Print ranked nodes in a readable format without embeddings."""
    for i, item in enumerate(ranked_nodes, start=1):
        node = item["node"]
        score = item["score"]

        print(f"🔹 **Result {i}**")
        print(f"📌 **Chunk ID:** {node['c.chunk_id']}")
        print(f"🗣 **Anförande Text:** {node['a.anforande_text'][:300]}...")  # Show only first 300 chars
        print(f"📜 **Chunk Text:** {node['c.text'][:300]}...")  # Show only first 300 chars
        print(f"⭐ **Similarity Score:** {score[0][0]:.4f}")  # Print similarity score with 4 decimals
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
    print_ranked_nodes(ranked_nodes=ranked_nodes)