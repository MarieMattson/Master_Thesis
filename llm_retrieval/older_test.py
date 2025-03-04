import os
from dotenv import load_dotenv
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USER")
NEO4J_PASSWORD =  os.getenv("NEO4J_PASSWORD")
OPEN_API_KEY = os.getenv("OPEN_API_KEY")
graph = Neo4jGraph()

graph.refresh_schema()
enhanced_graph = Neo4jGraph(enhanced_schema=True)

llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPEN_API_KEY, max_tokens=1500)
chain = GraphCypherQAChain.from_llm(
    graph=enhanced_graph, llm=llm, verbose=True, allow_dangerous_requests=True, return_intermediate_steps=True)

system_message = SystemMessage(
    content="""
        You're job is to respond to user queries about a debate from the Swedish parliament (riksdagen).
        The query and data will be in Swedish. 
        When generating a query, it must not be too specific. 
        For example, If asked about what Eva Flyborg said about a certain topic,
        You may retrieve all nodes related to Eva Flyborg.
        Also note that speakers's name are in caps, like "EVA FLYBORG"

        **Requirements:**
        - Only return a valid Cypher query—no explanations or summaries.
        - The speaker's name will be in uppercase with a party label (e.g., "JESSICA POLFJÄRD (M)").
        - The query should find the speaker’s "Anförande" nodes and related "Chunk" nodes.
        - The Protokoll ID will be provided (e.g., "H00998").
        - **Output ONLY the Cypher query.**
        - When generating the query, you must always include the `chunk_id` in the `RETURN` clause, along with the `text` and `anforande_text` of the nodes. 

        Example Cypher query format:
        MATCH (t:Talare {name: "EVA FLYBORG (FP)"}) 
        MATCH (t)-[:HALLER]->(a:Anforande) 
        MATCH (a)-[:HAS_CHUNK]->(c:Chunk)
        MATCH (t)-[:DELTAR_I]->(d:Debatt)-[:DOCUMENTED_IN]->(p:Protokoll {dok_id: "H0091"})
        RETURN a.anforande_text, c.text, c.chunk_id
    """
)


if __name__ == "__main__":
    query = (
        """
        Hur argumenterar JESSICA POLFJÄRD för sänkt restaurangmoms och fler jobb 
        i debatten från Protokoll H00998? Du MÅSTE returnera a.anforande_text, c.text, c.chunk_id, chunk.embedding!
        Generera en Cypher query som begränsar till den specifika debatten 
        och den aktuella talarens anförande, men undvik för många filter.
        """) 
    response = chain.invoke({
        "query": query#, 
        #"messages": [
            #system_message, 
        #    HumanMessage(content=query)
            #HumanMessage(content=" ".join(relevant_texts))  # Passing the relevant chunk texts as context
        #]
    })    
    print(response)