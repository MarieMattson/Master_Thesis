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
print(graph.schema)

graph.refresh_schema()
enhanced_graph = Neo4jGraph(enhanced_schema=True)

llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPEN_API_KEY)
chain = GraphCypherQAChain.from_llm(
    graph=enhanced_graph, llm=llm, verbose=True, allow_dangerous_requests=True,
    return_intermediate_steps=True)


if __name__ == "__main__":
    response = chain.invoke({"query": "Vilket parti har minst antal deltagare i debatten, och vilket har flest?"})
    print(graph.query("MATCH (p:Person) WHERE p.name = 'Gunnar Sträng' RETURN p AS Person"))
    print(response)