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
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD =  os.getenv("NEO4J_PASSWORD")

def retrieve_node(cypher_query:str)->list[str]:
    logger.info("Connecting to Neo4j at {} as {}",NEO4J_URI,NEO4J_USERNAME)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    cypher_query = cypher_query.replace('\n', ' ')
    try:
        with driver.session() as session:
                result = session.run(cypher_query)
                nodes = [record.data() for record in result]
                return nodes
    finally:
        driver.close()

if __name__ == "__main__":
    cypher_query ="""
                MATCH (t:Talare {name: "JESSICA POLFJÄRD"}) 
                MATCH (t)-[:HALLER]->(a:Anforande) 
                MATCH (a)-[:HAS_CHUNK]->(c:Chunk)
                MATCH (t)-[:DELTAR_I]->(d:Debatt)-[:DOCUMENTED_IN]->(p:Protokoll {dok_id: "H00998"})
                RETURN a.anforande_text, c.text, c.chunk_id, c.embedding        
                """
    print(retrieve_node(cypher_query))