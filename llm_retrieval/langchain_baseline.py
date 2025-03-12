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
chain = GraphCypherQAChain.from_llm(graph=enhanced_graph, 
                                    llm=llm, verbose=True, 
                                    return_intermediate_steps = True,
                                    allow_dangerous_requests=True, 
                                    validate_cypher=True)


if __name__ == "__main__":
    user_query = (
        """
        Hur argumenterar JESSICA POLFJÄRD för sänkt restaurangmoms och fler jobb 
        i debatten från Protokoll H00998? Du MÅSTE returnera a.anforande_text, c.text och c.chunk_id!
        Generera en Cypher query som begränsar till den specifika debatten
        Generera sedan ett svar baserat på texten du hämtar från cypher queryn.
        Svaret måste besvara frågan: Hur argumenterar JESSICA POLFJÄRD för sänkt restaurangmoms och fler jobb i debatten från Protokoll H00998?
        """) 
    response = chain.invoke({"query": user_query})
    print(response)
    print(30*"=","Langchain response", 30*"=")
    #print(response["result"])
    print(f"Final answer: {response['result']}")

