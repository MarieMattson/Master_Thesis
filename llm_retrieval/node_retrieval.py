'''
input: cypher query
output: neo4j node(s)
'''

import os
from dotenv import load_dotenv
from loguru import logger
from neo4j import GraphDatabase


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
                return nodes
                #names = [entry['p.name'] for entry in nodes] # names = [entry['Person']['name'] for entry in nodes]
                #return names
    finally:
        driver.close()

if __name__ == "__main__":
    print(retrieve_node('MATCH (person:Person)-[:BELONGS_TO]->(party:Party {partyName: "Centerpartiet"}) RETURN person.name, person.gender'))