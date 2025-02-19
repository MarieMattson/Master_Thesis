'''
input: natural language query
output: cypher query
'''
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

print(graph.query("MATCH (p:Person) WHERE p.name = 'Gunnar Sträng' RETURN p AS Person"))

graph.refresh_schema()
enhanced_graph = Neo4jGraph(enhanced_schema=True)
print(enhanced_graph)

llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPEN_API_KEY)
chain = GraphCypherQAChain.from_llm(
    graph=enhanced_graph, llm=llm, verbose=True, allow_dangerous_requests=True
)
response = chain.invoke({"query": "Vilket parti har minst antal deltagare i debatten, och vilket har flest?"})
print(response)

 
"""from llm_retrieval.openai_parses import OpenAIResponse

llm = ChatOpenAI("gpt-4o", model_provider="openai")
#embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


prompt_template = PromptTemplate(
    input_variables=["query"],
    template=(
        "You will get natural language queries in Swedish, and your job is to translate them into Cypher queries "
        "for a Neo4j database containing debates from the Swedish parliament. "
        "Ensure the queries follow Cypher conventions and formatting. Do not add new line characters. "
        "Only return a properly formatted Cypher query, nothing else. "
        "Ensure the query retrieves information for all aspects of the natural language query. \n\n"
        "### Database Schema: \n"
        "Nodes:\n"
        "- (:Person {{gender: man/woman, name: str, party: str}})\n"
        "- (:Statement {{text: str}})\n"
        "- (:Party {{partyName: str}})\n"
        "Relationships:\n"
        "- (:Person)-[:BELONGS_TO]->(:Party)\n"
        "- (:Person)-[:STATED]->(:Statement)\n\n"
        "### Example Translation:\n"
        "Natural Language Query: Vilket parti har flest kvinnor i debatten, och hur många kvinnor respektive män har de?\n"
        "Cypher Query:\n"
        "MATCH (person:Person)-[:BELONGS_TO]->(party:Party) "
        "WHERE person.gender = 'woman' "
        "WITH party, count(person) AS womenCount "
        "ORDER BY womenCount DESC "
        "LIMIT 1 "
        "MATCH (person:Person)-[:BELONGS_TO]->(party) "
        "WITH party.partyName AS partyName, womenCount, "
        "COUNT(CASE WHEN person.gender = 'man' THEN 1 END) AS menCount "
        "RETURN partyName, womenCount, menCount\n\n"
        "Now translate this query:\n{query}"
    ),
)


def translate(query: str)->str:
    messages = [
        SystemMessage(content="You are an AI that translates Swedish questions into Cypher queries."),
        HumanMessage(content=prompt_template.format(query=query))
    ]
    response = llm(messages)
    return response.content  
    '''api_key = os.getenv("OPEN_API_KEY")
    if not api_key:
        raise Exception("Sorry, not allowed to access chatbot...")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    body= {
            "model": 'gpt-4o', # Use the appropriate model
            "messages": [
            {
                "role": 'system',
                "content": "You will get natural language queries in Swedish, and you job is to translate to Cypher to navigate a Neo4j database containing debates from the Swedish parlament " +
                            "The queries must follow Cypher conventions and format, don't add new line characters, for instance"+
                            "Only return a properly formatted Cypher query, nothing else."+
                            "Make sure to retrieve something that can answer every part of the natural language query"+
                            "The following types of nodes exist in the database:"+
                            "(:Person {gender: man/woman, name: str, party: str})"+
                            "(:Statement {text: str})"+
                            "(:Party {partyName: str})"+
                            "The following types of relationships exist:"+
                            "(:Person)-[:BELONGS_TO]->(:Party)"
                            "(:Person)-[:STATED]->(:Statement)"+
                            "Here is an example of a good query translation from a natural language query:"+
                            "Vilket parti har flest kvinnor i debatten, och hur många kvinnor respektive män har de?"+
                            "MATCH (person:Person)-[:BELONGS_TO]->(party:Party) "+
                            "WHERE person.gender = 'woman' "+
                            "WITH party, count(person) AS womenCount "+
                            "ORDER BY womenCount DESC "+
                            "LIMIT 1 "+
                            "// Count men in the same party "+
                            "MATCH (person:Person)-[:BELONGS_TO]->(party) "+
                            "WITH party.partyName AS partyName, womenCount, "+
                            "    COUNT(CASE WHEN person.gender = 'man' THEN 1 END) AS menCount "+

                            "RETURN partyName, womenCount, menCount "+
                            "MATCH (p:Person)-[:BELONGS_TO]->(party:Party {partyName: 'Centerpartiet'}) "+ 
                            "RETURN p.name, p.gender"
                            },
            { "role": 'user', "content": query }
            ],
            "max_tokens": 300, # Adjust the token limit as needed
            "temperature": 0 # Adjust the creativity level as needed
        }

    response = requests.post(url, headers=headers, json=body)

    parsed_response = OpenAIResponse(**response.json())
    if len(parsed_response.choices) == 0:
        raise Exception("No answer")
    return parsed_response.choices[0].message.content'''



if __name__ == "__main__":
    response = translate("Vad är Frankrikes huvudstad?")
    print(response)"""