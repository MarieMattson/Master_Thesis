'''
input: cypher query
output: neo4j node(s)
'''

import os
from dotenv import load_dotenv
from loguru import logger
from neo4j import GraphDatabase

print("retrieval")

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD =  os.getenv("NEO4J_PASSWORD")

def retrieve_node(query:str)->list[str]:
    logger.info("Connecting to Neo4j at {} as {}",NEO4J_URI,NEO4J_USER)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
                result = session.run(query)
                nodes = [record.data() for record in result]
                names = [entry['person']['name'] for entry in nodes]
                return names
    finally:
        driver.close()

if __name__ == "__main__":
    retrieve_node('MATCH (p:Party {partyName: "Centerpartiet"})-[r:BELONGS_TO]-(person:Person) RETURN p, r, person')
    