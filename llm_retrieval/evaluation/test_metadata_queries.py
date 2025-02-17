from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
from llm_retrieval.wrapper import wrapper

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def run_query(query: str):
    """Execute a Cypher query and return results."""
    with driver.session() as session:
        result = session.run(query)
        return [record.data() for record in result]


def process_query(natural_language_query: str, cypher_query: str):
    """Executes a Cypher query, translates a natural language query, and prints results."""
    print(f" Natural Language Query: {natural_language_query}")
    try:
        result = run_query(cypher_query)
        llm_cypher_query, llm_result = wrapper(natural_language_query)

        print(f" Expected Cypher Query:\n{cypher_query.strip()}")
        print(f" Generated Cypher Query:\n{llm_cypher_query.strip()}")
        print(f" Neo4j Response: {result}")
        print(f" LLM Response: {llm_result}")
        print("-" * 80)
    except Exception as e:
        print(f"Something went wrong: {e}")

# List of queries to execute
queries = [
    {
        "natural_language_query": "Vilket parti har minst antal deltagare i debatten, och vilket har flest?",
        "cypher_query": """
        MATCH (person:Person)-[:BELONGS_TO]->(party:Party)
        WITH party.partyName AS partyName, count(person) AS participants
        ORDER BY participants ASC
        RETURN partyName, participants
        LIMIT 1

        UNION

        MATCH (person:Person)-[:BELONGS_TO]->(party:Party)
        WITH party.partyName AS partyName, count(person) AS participants
        ORDER BY participants DESC
        RETURN partyName, participants
        LIMIT 1
        """
    },
    {
        "natural_language_query": "Vem har flest inlägg i debatten och vilket parti tillhör personen?",
        "cypher_query": """
        MATCH (person:Person)-[:STATED]->()
        WITH person, count(person) AS statedCount
        ORDER BY statedCount DESC
        LIMIT 1
        RETURN person.name, statedCount, person.party
        """
    },
    {
        "natural_language_query": "Vilket parti har flest kvinnor i debatten, och hur många kvinnor respektive män har de?",
        "cypher_query": """
        MATCH (person:Person)-[:BELONGS_TO]->(party:Party)
        WHERE person.gender = 'woman'
        WITH party, count(person) AS womenCount
        ORDER BY womenCount DESC
        LIMIT 1

        // Count men in the same party
        MATCH (person:Person)-[:BELONGS_TO]->(party)
        WITH party.partyName AS partyName, womenCount, 
            COUNT(CASE WHEN person.gender = 'man' THEN 1 END) AS menCount

        RETURN partyName, womenCount, menCount
        """
    },
    {
        "natural_language_query": "Finns det något parti som inte har några kvinnliga deltagare i debatten?",
        "cypher_query": """
        MATCH (party:Party)
        WHERE NOT EXISTS {
            MATCH (party)<-[:BELONGS_TO]-(person:Person)
            WHERE person.gender = 'woman'
        }
        RETURN party.partyName
        """
    }
]

try:
    for query in queries:
        process_query(query["natural_language_query"], query["cypher_query"])
finally:
    driver.close()
