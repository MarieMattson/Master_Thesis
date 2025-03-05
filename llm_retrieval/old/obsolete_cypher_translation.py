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

llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)


def translate(query: str)->str:
    api_key = os.getenv("OPEN_API_KEY")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    body= {
            "model": 'gpt-4o-mini', # Use the appropriate model
            "messages": [
            {
                "role": 'system',
                "content": "Översätt användarens fråga till en cypher query för att söka i en Neo4j-databas med rikstdagsdebatter. Exempel på noder: " +
                            "(:Person {gender: 'man', name: 'John Eriksson', party: 'Centerpartiet'}), " +
                            "(:Party {partyName: 'Centerpartiet'}), " +
                            "(:Statement {text: 'Herr talman! ...'}). " +
                            "Exempel på relationer: " +
                            "'Person' -[:BELONGS_TO]-> 'Party', " +
                            "'Person' -[:STATED]-> 'Statement'. " +
                            "Returnera endast Cypher query, inget annat."+
                            "Returnera endast name och gender, inga andra properties"+
                            "Här är ett exempel på en bra query:"+
                            "MATCH (p:Person)-[:BELONGS_TO]->(party:Party {partyName: 'Centerpartiet'})"+ 
                            "RETURN p.name, p.gender"
                            },
            { "role": 'user', "content": query }
            ],
            "max_tokens": 100, # Adjust the token limit as needed
            "temperature": 0.7 # Adjust the creativity level as needed
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
        i debatten från Protokoll H00998? Du MÅSTE returnera a.anforande_text, c.text, c.chunk_id!
        Generera en Cypher query som begränsar till den specifika debatten 
        och den aktuella talarens anförande, men undvik för många filter.
        """) 
    response = translate(user_query)
    print(response)