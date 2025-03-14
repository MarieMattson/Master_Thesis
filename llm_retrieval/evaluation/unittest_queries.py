import json
from llm_retrieval.old import node_retrieval
from llm_retrieval.obsolete_full_pipeline import chain
import os
import unittest
from dotenv import load_dotenv
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI

from llm_retrieval.old.node_retrieval import retrieve_node

load_dotenv()


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
    },
    {
        "natural_language_query": "Hur många män respektive kvinnor finns i debatten",
        "cypher_query": """
        MATCH (person:Person)
        WHERE person.gender = 'woman'
        WITH count(person) AS womenCount

        MATCH (person:Person)
        WHERE person.gender = 'man'
        WITH womenCount, count(person) AS menCount

        RETURN womenCount, menCount
        """
    },
    {
        "natural_language_query": "Vilka deltagare säger bara en sak under debatten?",
        "cypher_query": """
        MATCH (p:Person)-[:STATED]->(s:Statement)
        WITH p, COUNT(s) AS statementCount
        WHERE statementCount = 1
        RETURN p.name, statementCount
        """
    },
    {
        "natural_language_query": "Finns det några deltagare i debatten som har samma förnamn?",
        "cypher_query": """
        MATCH (p:Person)
        WITH SPLIT(p.name, " ")[0] AS firstName, COUNT(p) AS nameCount
        WHERE nameCount > 1
        RETURN firstName, nameCount
        ORDER BY nameCount DESC
        """
    },
    {
        "natural_language_query": "Hur många partier deltog i debatten?",
        "cypher_query": """
        MATCH (p:Party)
        RETURN COUNT(p) AS numberOfParties
        """
    }
]   

class TestGraphCypherChain(unittest.TestCase):

    
    def test_retrieved_nodes(self):
        for query_data in queries:
            with self.subTest(query=query_data["natural_language_query"]):
                expected_nodes = retrieve_node(query_data["cypher_query"]) 
                llm_response = chain.invoke({"query": query_data["natural_language_query"]})["intermediate_steps"][1]["context"]

                try:
                    self.assertEqual(
                        set(map(str, expected_nodes)),
                        set(map(str, llm_response)),
                        f"Mismatch for query: {query_data['natural_language_query']}\nExpected Nodes: {expected_nodes}\nGenerated Nodes: {llm_response}"
                    )
                except AssertionError as e:
                    print("\n=== Cypher Query for Mismatch ===")
                    print(query_data["cypher_query"])
                    print("\n" + "=" * 50 + "\n")
                    print("\n=== Expected nodes ===")
                    print(expected_nodes)
                    raise e



if __name__ == "__main__":
    unittest.main()
