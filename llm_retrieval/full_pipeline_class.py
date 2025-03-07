import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger
from neo4j import GraphDatabase
import numpy as np
import requests
import sklearn
from langchain_core.messages import HumanMessage, SystemMessage
from llm_retrieval.openai_parses import OpenAIResponse
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD =  os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
graph = Neo4jGraph()
graph.refresh_schema()
enhanced_graph = Neo4jGraph(enhanced_schema=True)


class GraphRAG():
    def __init__(self):
        self.url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        self.llm = ChatOpenAI(model="gpt-4")
        self.embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

        
    def translate_to_cypher(self, query: str)->str:
        system_prompt = """
                        You're job is to respond to user queries about a debate from the Swedish parliament (riksdagen).
                        The query and data will be in Swedish. 
                        You should filter out everything that is relevant for the speaker,
                        for example, If asked about what Eva Flyborg said about a certain topic,
                        You may retrieve all nodes related to Eva Flyborg.
                        Second, you shold filter out the nodes that have the highest similarity score to the user query        
                        Also note that speakers's name are in caps, like "EVA FLYBORG"

                        **Graph Schema:**
                        {enhanced_graph}

                        **Requirements:**
                        - DO NOT ADD ANY NEW LINES OR ANY WRITING EXCEPT THE CYPHER QUERY
                        - DO NOT ADD ANYTHING TO THE NAMES, INCLUDING PARTY ASSOCIATION
                        - Only return a valid Cypher query—no explanations or summaries.
                        - The speaker's name will be in uppercase with a party label (e.g., "JESSICA POLFJÄRD").
                        - The query should find the speaker’s "Anförande" nodes and related "Chunk" nodes.
                        - The Protokoll ID will be provided (e.g., "H00998").
                        - **Output ONLY the Cypher query.**
                        - When generating the query, you must always include the `chunk_id` in the `RETURN` clause, along with the `text` and `anforande_text` of the nodes. 

                        Example Cypher query format:
                        MATCH (t:Talare {name: "EVA FLYBORG"}) MATCH (t)-[:HALLER]->(a:Anforande) MATCH (a)-[:HAS_CHUNK]->(c:Chunk) MATCH (t)-[:DELTAR_I]->(d:Debatt)-[:DOCUMENTED_IN]->(p:Protokoll {dok_id: "H0091"}) RETURN a.anforande_text, c.text, c.chunk_id
                        """
        body = {
                "model": 'gpt-4o',
                "messages": [
                    {"role": 'system', "content": system_prompt},
                    {"role": 'user', "content": query}
                ],
                "max_tokens": 150,
                "temperature": 0
            }
        
        response = requests.post(self.url, headers=self.headers, json=body)

        parsed_response = OpenAIResponse(**response.json())
        if len(parsed_response.choices) == 0:
            raise Exception("No answer")
        return parsed_response.choices[0].message.content
    
    @staticmethod
    def retrieve_nodes(cypher_query:str)->list[str]:
        #logger.info("Connecting to Neo4j at {} as {}",NEO4J_URI,NEO4J_USERNAME)
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        cypher_query = cypher_query.replace('\n', ' ')
        try:
            with driver.session() as session:
                    result = session.run(cypher_query)
                    nodes = [record.data() for record in result]
                    return nodes
        finally:
            driver.close()
    
    def rank_nodes_by_similarity(self, query_text: str, retrieved_nodes: list[dict], top_k=6) -> list[dict]:
        """Rank retrieved nodes based on cosine similarity with the user query."""
        
        query_embedding = np.array(self.embedding.embed_query(query_text)).reshape(1, -1)

        ranked_nodes = []
        for node in retrieved_nodes:
            chunk_embedding = node.get('c.embedding', None)
            if chunk_embedding:
                chunk_embedding = np.array(chunk_embedding).reshape(1,-1)
                similarity = sklearn.metrics.pairwise.cosine_similarity(query_embedding, chunk_embedding)
                ranked_nodes.append((node, similarity))

        ranked_nodes = sorted(ranked_nodes, key=lambda x: x[1], reverse=True)

        return [{"node": item[0], "score": item[1]} for item in ranked_nodes[:top_k]]

    @staticmethod
    def print_ranked_nodes(ranked_nodes:list[dict]):
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


    def generate_response(self, ranked_nodes:list[dict], user_query:str, top_k=3):
        
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
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        return response.content

if __name__ == "__main__":
    graph_rag = GraphRAG()

    user_query = (
        """
        Hur argumenterar JESSICA POLFJÄRD för sänkt restaurangmoms och fler jobb 
        i debatten från Protokoll H00998? Du MÅSTE returnera a.anforande_text, c.text, c.chunk_id och c.embedding!
        Generera en Cypher query som begränsar till den specifika debatten 
        och den aktuella talarens anförande, men undvik för många filter.
        """) 
        
    print("Translating user query into Cypher query...")
    cypher_query = graph_rag.translate_to_cypher(user_query)
    print("Generated Cypher Query:", cypher_query)

    print("\nRetrieving nodes from Neo4j...")
    retrieved_nodes = graph_rag.retrieve_nodes(cypher_query)
    print(f"Retrieved {len(retrieved_nodes)} nodes.")

    print("\nRanking nodes by similarity...")
    ranked_nodes = graph_rag.rank_nodes_by_similarity(user_query, retrieved_nodes)
    
    print("\nPrinting ranked nodes...")
    graph_rag.print_ranked_nodes(ranked_nodes)

    print("\nGenerating final response...")
    final_response = graph_rag.generate_response(ranked_nodes, user_query)
    print("\nFinal Response:\n", final_response)
