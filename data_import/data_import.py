from neo4j import GraphDatabase
from dotenv import load_dotenv
import pandas as pd
from loguru import logger
import os

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD =  os.getenv("NEO4J_PASSWORD")
logger.info("Connecting to Neo4j at {} as {}",NEO4J_URI,NEO4J_USER)



def test_connection():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            result = session.run("RETURN 'Hello, Neo4j!' AS message")
            for record in result:
                print(record["message"])
    finally:
        driver.close()


def load_csv(file_path):
    df = pd.read_csv(file_path)
    return df


def insert_data_into_neo4j(df):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            for _, row in df.iterrows():
                session.run(
                    """
                    MERGE (p:Person {name: $name, party: $party, gender: $gender})
                    CREATE (s:Statement {text: $text})
                    MERGE (p)-[:STATED]->(s)
                    """,
                    #id=row["id"],
                    name=row["name"],
                    party=row["party"],
                    gender=row["gender"],
                    #start_segment=row["start_segment"],
                    #end_segment=row["end_segment"],
                    text=row["text"]
                )
    finally:
        driver.close()

if __name__ == "__main__":
    test_connection()  # Should print: Hello, Neo4j!
    file_path = "data_import/data/filtered_riksdag.csv"
    df = load_csv(file_path)
    insert_data_into_neo4j(df)
    print("Data successfully imported into Neo4j!")
