import json
from llm_retrieval.full_pipeline import chain
import os
import unittest
from dotenv import load_dotenv
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI

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
]

class TestGraphCypherChain(unittest.TestCase):

    def test_generated_queries(self):
        for query_data in queries:
            with self.subTest(query=query_data["natural_language_query"]):
                response = chain.invoke({"query": query_data["natural_language_query"]})
                intermediate_steps = response.get("intermediate_steps", [])
                generated_cypher = ""
                for step in intermediate_steps:
                    if "query" in step:
                        generated_cypher = step["query"].strip()
                        break

                expected_cypher = query_data["cypher_query"].strip()
               
                print("\n=== Expected Cypher Query ===")
                print(expected_cypher)
                print("\n=== Generated Cypher Query ===")
                print(generated_cypher)
                print("\n" + "=" * 50 + "\n")  

if __name__ == "__main__":
    unittest.main()
