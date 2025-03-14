from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1500,   # Max 1500 characters per chunk
    chunk_overlap=200  # 200 characters overlap to maintain context
)

def fetch_all_anforande():
    with driver.session() as session:
        result = session.run(
            """
            MATCH (p:Protokoll)
            OPTIONAL MATCH (d:Debatt)-[:DOCUMENTED_IN]->(p)
            OPTIONAL MATCH (t:Talare)-[:DELTAR_I]->(d)
            OPTIONAL MATCH (t)-[:HALLER]->(a:Anforande)
            RETURN p.dok_id, a.anforande_id, a.anforande_text
            """
        )
        data = {}
        for record in result:
            dok_id = record["p.dok_id"]
            anforande_id = record["a.anforande_id"]
            anforande_text = record["a.anforande_text"]
            if dok_id and anforande_id and anforande_text:
                if dok_id not in data:
                    data[dok_id] = []
                data[dok_id].append({"anforande_id": anforande_id, "text": anforande_text})
        return data

def store_chunks_in_neo4j(anforande_id, chunks):
    with driver.session() as session:
        for i, chunk in enumerate(chunks):
            embedding = embeddings_model.embed_query(chunk)  # Generate embedding
            
            session.run(
                """
                MATCH (a:Anforande {anforande_id: $anforande_id})
                CREATE (c:Chunk {
                    chunk_id: apoc.create.uuid(),
                    chunk_index: $chunk_index,
                    text: $text,
                    embedding: $embedding
                })
                MERGE (a)-[:HAS_CHUNK]->(c)
                """,
                anforande_id=anforande_id,
                chunk_index=i,
                text=chunk,
                embedding=embedding
            )

data = fetch_all_anforande()
if not data:
    print("No speeches found in the database.")
else:
    for dok_id, anforande_list in data.items():
        print(f"Found {len(anforande_list)} Anforande(s) for Protokoll {dok_id}.")
        
        for item in anforande_list:
            chunks = text_splitter.split_text(item["text"])
            print(f"Generated {len(chunks)} chunks for Anforande {item['anforande_id']}.")
            store_chunks_in_neo4j(item["anforande_id"], chunks)

    print("All chunks stored successfully.")

driver.close()


'''
def fetch_anforande_for_protokoll(dok_id):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (p:Protokoll {dok_id: $dok_id})
            OPTIONAL MATCH (d:Debatt)-[:DOCUMENTED_IN]->(p)
            OPTIONAL MATCH (t:Talare)-[:DELTAR_I]->(d)
            OPTIONAL MATCH (t)-[:HALLER]->(a:Anforande)
            RETURN a.anforande_id, a.anforande_text
            """,
            dok_id=dok_id,
        )
        return [
            {"anforande_id": record["a.anforande_id"], "text": record["a.anforande_text"]}
            for record in result if record["a.anforande_id"] and record["a.anforande_text"]
        ]

def store_chunks_in_neo4j(anforande_id, chunks):
    with driver.session() as session:
        for i, chunk in enumerate(chunks):
            embedding = embeddings_model.embed_query(chunk)  # Generate embedding
            
            session.run(
                """
                MATCH (a:Anforande {anforande_id: $anforande_id})
                CREATE (c:Chunk {
                    chunk_id: apoc.create.uuid(),
                    chunk_index: $chunk_index,
                    text: $text,
                    embedding: $embedding
                })
                MERGE (a)-[:HAS_CHUNK]->(c)
                """,
                anforande_id=anforande_id,
                chunk_index=i,
                text=chunk,
                embedding=embedding
            )

# Run for a specific Protokoll (e.g., H00998)
#DOK_ID = "H00998"

data = fetch_anforande_for_protokoll(DOK_ID)
if not data:
    print(f"No speeches found for Protokoll {DOK_ID}.")
else:
    print(f"Found {len(data)} Anforande(s) for Protokoll {DOK_ID}.")

    for item in data:
        chunks = text_splitter.split_text(item["text"])
        print(f"Generated {len(chunks)} chunks for Anforande {item['anforande_id']}.")
        store_chunks_in_neo4j(item["anforande_id"], chunks)

    print(f"All chunks stored successfully for Protokoll {DOK_ID}.")

driver.close()
'''