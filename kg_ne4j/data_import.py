from neo4j import GraphDatabase
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
print(f"Connecting to Neo4j at {NEO4J_URI} as {NEO4J_USER}")



# Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Test the connection
def test_connection():
    with driver.session() as session:
        result = session.run("RETURN 'Hello, Neo4j!' AS message")
        for record in result:
            print(record["message"])

test_connection()  # Should print: Hello, Neo4j!

# Close connection when done
driver.close()


def load_csv(file_path):
    df = pd.read_csv(file_path)
    return df


def insert_data_into_neo4j(df):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
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
    driver.close()


file_path = "thesis/filtered_riksdag.csv"
df = load_csv(file_path)
insert_data_into_neo4j(df)
print("Data successfully imported into Neo4j!")
