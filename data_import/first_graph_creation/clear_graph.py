from neo4j import GraphDatabase
from dotenv import load_dotenv
import pandas as pd
from loguru import logger
import os

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USER")
NEO4J_PASSWORD =  os.getenv("NEO4J_PASSWORD")
logger.info("Connecting to Neo4j at {} as {}",NEO4J_URI,NEO4J_USERNAME)

def clear_graph():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Graph cleared")
    finally:
        driver.close()

clear_graph() # Should print: Graph cleared