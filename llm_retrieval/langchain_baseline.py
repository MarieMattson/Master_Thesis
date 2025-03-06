import os
from dotenv import load_dotenv
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USER")
NEO4J_PASSWORD =  os.getenv("NEO4J_PASSWORD")
OPEN_API_KEY = os.getenv("OPEN_API_KEY")
graph = Neo4jGraph()

graph.refresh_schema()
enhanced_graph = Neo4jGraph(enhanced_schema=True)

llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPEN_API_KEY)
chain = GraphCypherQAChain.from_llm(
    graph=enhanced_graph, llm=llm, verbose=True, allow_dangerous_requests=True, validate_cypher=True)


if __name__ == "__main__":
    response = chain.invoke({"query": "Vad tycker BIRGITTA OHLSSON om skatt?"})

    print(response) 