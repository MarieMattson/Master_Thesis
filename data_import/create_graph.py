from neo4j import GraphDatabase
from dotenv import load_dotenv
import pandas as pd
from loguru import logger
import os
import json

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
logger.info("Connecting to Neo4j at {} as {}", NEO4J_URI, NEO4J_USERNAME)


def test_connection():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            result = session.run("RETURN 'Hello, Neo4j!' AS message")
            for record in result:
                print(record["message"])
    finally:
        driver.close()

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def insert_data_into_neo4j(data):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            for row in data:
                session.run("""
                    MERGE (p:Protokoll {dok_hangar_id: $dok_hangar_id, dok_id: $dok_id})
                    ON CREATE SET p.dok_titel = $dok_titel, p.dok_rm = $dok_rm, p.dok_datum = $dok_datum

                    MERGE (d:Debatt {avsnittsrubrik: $avsnittsrubrik, kammaraktivitet: COALESCE($kammaraktivitet, 'Unknown')})

                    MERGE (t:Talare {name: $talare, party: COALESCE($parti, 'Unknown')})

                    MERGE (a:Anforande {anforande_id: $anforande_id, anforande_nummer: $anforande_nummer})
                    ON CREATE SET a.replik = $replik, a.anforande_text = $anforande_text

                    MERGE (d)-[:DOCUMENTED_IN]->(p)
                    MERGE (t)-[:DELTAR_I]->(d)
                    MERGE (t)-[:HALLER]->(a)
                    """,
                    dok_hangar_id=row["dok_hangar_id"],
                    dok_id=row["dok_id"],
                    dok_titel=row["dok_titel"],
                    dok_rm=row["dok_rm"],
                    dok_datum=row["dok_datum"],
                    avsnittsrubrik=row["avsnittsrubrik"],
                    kammaraktivitet=row["kammaraktivitet"],
                    talare=row["talare"],
                    parti=row["parti"],
                    anforande_id=row["anforande_id"],
                    anforande_nummer=row["anforande_nummer"],
                    replik=row["replik"],
                    anforande_text=row["anforandetext"],
                    intressent_id=row["intressent_id"],
                    rel_dok_id=row["rel_dok_id"],
                    underrubrik=row["underrubrik"]
                )

    finally:
        driver.close()

if __name__ == "__main__":
    test_connection()  # Should print: Hello, Neo4j!
    file_path = "/mnt/c/Users/User/thesis/data_import/filtered_riksdag_exp1.json"
    data = load_json(file_path)
    insert_data_into_neo4j(data)
    print("Data successfully imported into Neo4j!")
