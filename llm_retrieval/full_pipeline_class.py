import os
from xmlrpc.client import boolean
from langchain.schema import Document
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger
from neo4j import GraphDatabase
import numpy as np
import requests
import sklearn
from nltk.tokenize import word_tokenize
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import Boolean
from llm_retrieval.old.openai_parses import OpenAIResponse
from langchain_community.retrievers import BM25Retriever

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD =  os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
graph = Neo4jGraph()
graph.refresh_schema()
enhanced_graph = graph.get_structured_schema #Neo4jGraph(enhanced_schema=True)
print(enhanced_graph)

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
        system_prompt = f"""
        You are a useful assistans whose job is to generate cypher queries based on user queries about debates from the Swedish parliament (riksdagen).
        The query and data will be in Swedish. 

        **Graph Schema:**
        {enhanced_graph}

        **Requirements:**
        - Only use metadata in your cypher queries, not the contents of the anforande_text
        - DO NOT ADD ANY NEW LINES OR ANY WRITING EXCEPT THE CYPHER QUERY
        - DO NOT ADD ANYTHING TO THE NAMES, INCLUDING PARTY ASSOCIATION
        - Only return a valid Cypher query—no explanations or summaries.
        - The query should find the speaker’s "Anförande" nodes and related "Chunk" nodes.
        - **Output ONLY the Cypher query.**
        - You must **ALWAYS** include the `a.anforande_text, c.text, c.chunk_id, c.embedding, a.anforande_id` in the `RETURN` clause!. 
        - You must always RETURN DISTINCT to avoid duplicates

        You should filter out everything that is relevant for the speaker or party,
        for example, If asked about what Eva Flyborg said about a certain topic you may retrieve all nodes related to Eva Flyborg.
        
        The question and answer will be written in **Swedish**.
        In the data, the party association is written as the short form, but in the questions, they **will be written out**.
        **ALWAYS use the short form when querying the database**
        M: Moderaterna, S: Socialdemokraterna, SD: Sverigedemokraterna, C: Centerpartiet, V: Vänsterpartiet, L: Liberalerna, KD: Kristdemokraterna, MP: Miljöpartiet de Gröna

        If the questions ask about a specific time period, the month will be written out, like "februari 2022" or "december 2019"
        However in the data, dates are written like 2022-05-23".
        Dates are **ONLY** a property in the protocol node 


        Example Cypher query format:
        To retrieve all speeches from a person: MATCH (t:Talare {{name: "Eva Flyborg"}}) MATCH (t)-[:HALLER]->(a:Anforande) MATCH (a)-[:HAS_CHUNK]->(c:Chunk) MATCH (t)-[:DELTAR_I]->(d:Debatt)-[:DOCUMENTED_IN]->(p:Protokoll) RETURN DISTINCT a.anforande_text, c.text, c.chunk_id, c.embedding
        To retrieve all speeches from a party: MATCH (t:Talare {{party: "M"}}) MATCH (t)-[:HALLER]->(a:Anforande) MATCH (a)-[:HAS_CHUNK]->(c:Chunk) MATCH (t)-[:DELTAR_I]->(d:Debatt)-[:DOCUMENTED_IN]->(p:Protokoll) RETURN DISTINCT a.anforande_text, c.text, c.chunk_id, c.embedding
        To retrieve all speeches from a person during a particular month: MATCH (t:Talare {{name: "Eva Flyborg"}}) MATCH (t)-[:HALLER]->(a:Anforande) MATCH (a)-[:HAS_CHUNK]->(c:Chunk) MATCH (t)-[:DELTAR_I]->(d:Debatt)-[:DOCUMENTED_IN]->(p:Protokoll) WHERE p.dok_datum CONTAINS "2020-01" RETURN DISTINCT a.anforande_text, c.text, c.chunk_id, c.embedding
        To retrieve all speeches from a party during a particular month: MATCH (t:Talare{{party:"M"}}) MATCH (t)-[:HALLER]->(a:Anforande) MATCH (a)-[:HAS_CHUNK]->(c:Chunk) MATCH (t)-[:DELTAR_I]->(d:Debatt)-[:DOCUMENTED_IN]->(p:Protokoll) WHERE p.dok_datum CONTAINS "2020-01" RETURN DISTINCT a.anforande_text, c.text, c.chunk_id, c.embedding
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
                similarity = sklearn.metrics.pairwise.cosine_similarity(query_embedding, chunk_embedding).tolist()
                ranked_nodes.append((node, similarity))
            if chunk_embedding is None:
                logger.warning(f"Missing embedding for node: {node['c.chunk_id']}")
                continue


        ranked_nodes = sorted(ranked_nodes, key=lambda x: x[1], reverse=True)

        return [{"node": item[0], "score": item[1]} for item in ranked_nodes[:top_k]]

    def rank_nodes_with_BM25(self, query_text: str, retrieved_nodes: list[dict], top_k=6)->list[dict]:
        # Convert each node to a LangChain Document
        documents = [
            Document(
                page_content=node['c.text'],
                metadata={"original_node": node}

            )
            for node in retrieved_nodes
        ]

        retriever = BM25Retriever.from_documents(documents, k=top_k,preprocess_func=word_tokenize)

        results = retriever.invoke(query_text)
        return [{"node": doc.metadata["original_node"], "score": None} for doc in results]


    @staticmethod
    def print_ranked_nodes(ranked_nodes:list[dict], is_cosine:boolean):
        """Print ranked nodes in a readable format without embeddings."""
        for i, item in enumerate(ranked_nodes, start=1):
            node = item["node"]
            score = item["score"]

            print(f"🔹 **Result {i}**")
            print(f"🗣 **Anförande Text:** {node['a.anforande_text'][:300]}...")
            print(f"📜 **Chunk Text:** {node['c.text']}...")
            if is_cosine:
                print(f"⭐ **Similarity Score:** {score[0][0]:.4f}")  
            print("-" * 80)


    def generate_response(self, ranked_nodes:list[dict], user_query:str):
        
        prompt = f"User query: {user_query}\n\nBased on the ranked debate chunks, generate a relevant response to the user query using the following information You must respond in Swedish:\n\n"
        for i, item in enumerate(ranked_nodes, start=1):
            node = item["node"]
            prompt += f"\n🔹 **Result {i}:**\n"
            prompt += f"🗣 **Anförande Text:** {node['a.anforande_text']}...\n"
        prompt += "\nNow, generate a response based on the context provided above. Make sure to answer the user query in a natural and coherent way based on the information from the debate. You must respond in Swedish"
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        return response.content

if __name__ == "__main__":
    graph_rag = GraphRAG()
    system_prompt = f"""
                        You are a useful assistans whose job is to generate cypher queries based on user queries about debates from the Swedish parliament (riksdagen).
                        The Cypher query will be used to query and external database.
                        The query and data will be in Swedish. 

                        **Graph Schema:**
                        {enhanced_graph}

                        **Requirements:**
                        - Only use metadata in your cypher queries, not the contents of the anforande_text
                        - DO NOT ADD ANY NEW LINES OR ANY WRITING EXCEPT THE CYPHER QUERY
                        - DO NOT ADD ANYTHING TO THE NAMES, INCLUDING PARTY ASSOCIATION
                        - Only return a valid Cypher query—no explanations or summaries.
                        - The query should find the speaker’s "Anförande" nodes and related "Chunk" nodes.
                        - **Output ONLY the Cypher query.**
                        - When generating the query, you must always include the `c.anforande_id, a.anforande_text, c.text, c.embeddin` in the `RETURN` clause

                        You should filter out everything that is relevant for the speaker or party
                        for example, If asked about what Eva Flyborg said about a certain topic you may retrieve all nodes related to Eva Flyborg.
                        If asked about the stance of Moderaterna in a certain topic, you should return everything from M (Moderaterna) 

                        Example Cypher query format:
                        MATCH (t:Talare {{name: "Eva Flyborg"}}) MATCH (t)-[:HALLER]->(a:Anforande) MATCH (a)-[:HAS_CHUNK]->(c:Chunk) MATCH (t)-[:DELTAR_I]->(d:Debatt)-[:DOCUMENTED_IN]->(p:Protokoll) RETURN a.anforande_text, c.text, c.chunk_id
                        MATCH (t:Talare {{party: "M"}}) MATCH (t)-[:HALLER]->(a:Anforande) MATCH (a)-[:HAS_CHUNK]->(c:Chunk) MATCH (t)-[:DELTAR_I]->(d:Debatt)-[:DOCUMENTED_IN]->(p:Protokoll) RETURN a.anforande_text, c.text, c.chunk_id
                        """
    print(system_prompt)

    user_query = (
        """
        Vad är Socialdemokraternas position i frågan om hanteringen av kris- och återstartsmedlen?
        """) 
      
    print("Translating user query into Cypher query...")
    cypher_query = graph_rag.translate_to_cypher(user_query)
    print("Generated Cypher Query:", cypher_query)

    print("\nRetrieving nodes from Neo4j...")
    retrieved_nodes = graph_rag.retrieve_nodes(cypher_query)
    print(f"Retrieved {len(retrieved_nodes)} nodes.")

    print("\nRanking nodes by similarity...")
    cosine_ranked_nodes = graph_rag.rank_nodes_by_similarity(user_query, retrieved_nodes)
    
    print("\nPrinting cosine ranked nodes...")
    graph_rag.print_ranked_nodes(cosine_ranked_nodes, is_cosine=True)

    print("\nRanking nodes with BM25...")
    bm25_ranked_nodes = graph_rag.rank_nodes_with_BM25(user_query, retrieved_nodes)
    print("\nPrinting BM25 ranking")
    graph_rag.print_ranked_nodes(bm25_ranked_nodes, is_cosine=False)

    print("\nGenerating final response...")
    print("\nCosine version")
    final_response = graph_rag.generate_response(cosine_ranked_nodes, user_query)
    print("\nFinal Response:\n", final_response)
    
    print("\nBM25 version")
    final_response = graph_rag.generate_response(bm25_ranked_nodes, user_query)
    print("\nFinal Response:\n", final_response)