'''
input: natural language query
output: cypher query
'''
import os
import requests

from llm_retrieval.openai_parses import OpenAIResponse

def translate(query: str)->str:
    api_key = os.getenv("OPEN_API_KEY")
    if not api_key:
        raise Exception("Sorry, not allowed to access chatbot...")
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

    parsed_response = OpenAIResponse(**response.json())
    if len(parsed_response.choices) == 0:
        raise Exception("No answer")
    return parsed_response.choices[0].message.content



if __name__ == "__main__":
    response = translate("Vad är Frankrikes huvudstad?")
    print(response)