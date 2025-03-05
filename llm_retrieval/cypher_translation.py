'''
input: natural language query
output: cypher query
'''
import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
import requests

from llm_retrieval.openai_parses import OpenAIResponse


load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USER")
NEO4J_PASSWORD =  os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
graph = Neo4jGraph()

graph.refresh_schema()
enhanced_graph = Neo4jGraph(enhanced_schema=True)


def translate(query: str)->str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
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
                    - Only return a valid Cypher query—no explanations or summaries.
                    - The speaker's name will be in uppercase with a party label (e.g., "JESSICA POLFJÄRD (M)").
                    - The query should find the speaker’s "Anförande" nodes and related "Chunk" nodes.
                    - The Protokoll ID will be provided (e.g., "H00998").
                    - **Output ONLY the Cypher query.**
                    - When generating the query, you must always include the `chunk_id` in the `RETURN` clause, along with the `text` and `anforande_text` of the nodes. 

                    Example Cypher query format:
                    MATCH (t:Talare {name: "EVA FLYBORG"}) 
                    MATCH (t)-[:HALLER]->(a:Anforande) 
                    MATCH (a)-[:HAS_CHUNK]->(c:Chunk)
                    MATCH (t)-[:DELTAR_I]->(d:Debatt)-[:DOCUMENTED_IN]->(p:Protokoll {dok_id: "H0091"})
                    RETURN a.anforande_text, c.text, c.chunk_id
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

    response = requests.post(url, headers=headers, json=body)

    #print(response.json())
    parsed_response = OpenAIResponse(**response.json())
    if len(parsed_response.choices) == 0:
        raise Exception("No answer")
    return parsed_response.choices[0].message.content



if __name__ == "__main__":
    user_query = (
        """
        Hur argumenterar JESSICA POLFJÄRD för sänkt restaurangmoms och fler jobb 
        i debatten från Protokoll H00998? Du MÅSTE returnera a.anforande_text, c.text, c.chunk_id och c.embedding!
        Generera en Cypher query som begränsar till den specifika debatten 
        och den aktuella talarens anförande, men undvik för många filter.
        """) 
    response = translate(user_query)
    print(response)