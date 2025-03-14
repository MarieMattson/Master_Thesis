"""Potential improvement:
    If no response is found, it should be possible to have it generate a response based on other 
    people's statements. This could be useful if no response is found directly in the speakers anforanden.
    However, this should be clearly separated and the system should make it clear from which part the response comes.
"""
import os
from dotenv import load_dotenv
from langchain.schema import SystemMessage, HumanMessage
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from llm_retrieval.old.node_ranking import rank_nodes_by_similarity
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



def response_generation(ranked_nodes, user_query, top_k=3):
    
    top_nodes = ranked_nodes[:top_k]
    
    prompt = f"User query: {user_query}\n\nBased on the ranked debate chunks, generate a relevant response to the user query using the following information You must respond in Swedish:\n\n"
    for i, item in enumerate(top_nodes, start=1):
        node = item["node"]
        prompt += f"\n🔹 **Result {i}:**\n"
        prompt += f"📌 **Chunk ID:** {node['c.chunk_id']}\n"
        prompt += f"🗣 **Anförande Text:** {node['a.anforande_text'][:300]}...\n"
        prompt += f"📜 **Chunk Text:** {node['c.text']}...\n" 
        prompt += f"⭐ **Similarity Score:** {item['score'][0][0]:.4f}\n"
    prompt += "\nNow, generate a response based on the context provided above. Make sure to answer the user query in a natural and coherent way based on the information from the debate. You must respond in Swedish"
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return response.content

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
    response = response_generation(ranked_nodes, user_query)
    print("\nGenerated Response:")
    print(response)
