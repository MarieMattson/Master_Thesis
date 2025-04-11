from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import hashlib

dok_ids = ['H90965', 'H909139', 'H8099', 'H90995', 'H80933', 'H9095', 'H90992', 'H80928', 'H909112', 'H80935', 'H80988', 'H80992', 'H909137', 'H8096', 'H90958', 'H90952', 'H80955', 'H80942', 'H809113', 'H80971', 'H8097', 'H90941', 'H80969', 'H80914', 'H909116', 'H90971', 'H9094', 'H80910', 'H90923', 'H809139', 'H80927', 'H80961', 'H809130', 'H80948', 'H80939', 'H80968', 'H90978', 'H809117', 'H90921', 'H809137', 'H80979', 'H909102', 'H80985', 'H90933', 'H9091', 'H909101', 'H809153', 'H90924', 'H90953', 'H90948', 'H809124', 'H909110', 'H9093', 'H90967', 'H80963', 'H80975', 'H909107', 'H80950', 'H80978', 'H80956', 'H80940', 'H9099', 'H809144', 'H909117', 'H909122', 'H809123', 'H80966', 'H909123', 'H909121', 'H90922', 'H90989', 'H90990', 'H80954', 'H80980', 'H909142', 'H809116', 'H809142', 'H90998', 'H80981', 'H909106', 'H80977', 'H90938', 'H90996', 'H80999', 'H80937', 'H90979', 'H80920', 'H80996', 'H809107', 'H909131', 'H909118', 'H80997', 'H90988', 'H809114', 'H909100', 'H90919', 'H90930', 'H80983', 'H90966', 'H80952', 'H809105', 'H90918', 'H909111', 'H809140', 'H809134', 'H809101', 'H90920', 'H90973', 'H809135', 'H80947', 'H90950', 'H809104', 'H80991', 'H90939', 'H80953', 'H80974', 'H809145', 'H90913', 'H8095', 'H90917', 'H809119', 'H80934', 'H80998', 'H809158', 'H809157', 'H809115', 'H80915', 'H809146', 'H809106', 'H8092', 'H8098', 'H80993', 'H90993', 'H90963', 'H80949', 'H8094', 'H90960', 'H80982', 'H809121', 'H90951', 'H809122', 'H80929', 'H809127', 'H80932', 'H90942', 'H909120', 'H80925', 'H90991', 'H809103', 'H80964', 'H80972', 'H809138', 'H90982', 'H809136', 'H80967', 'H809112', 'H909134', 'H80986', 'H80924', 'H809147', 'H90956', 'H80965', 'H90959', 'H80921', 'H90943', 'H90912', 'H909128', 'H90947', 'H8093', 'H90972', 'H90975', 'H80987', 'H909141', 'H909138', 'H809126', 'H80990', 'H909113', 'H90946', 'H90914', 'H80962', 'H80918', 'H80917', 'H80938', 'H809111', 'H80944', 'H809141', 'H80936', 'H909104', 'H90934', 'H90910', 'H809109', 'H80945', 'H909133', 'H80973', 'H80930', 'H909114', 'H80919', 'H9098', 'H909999', 'H809110', 'H809150', 'H809155', 'H90969', 'H90987', 'H80926', 'H90968', 'H809108', 'H809152', 'H80951', 'H809125', 'H80957', 'H9097', 'H909135', 'H90994', 'H90999', 'H90926', 'H809120', 'H909127', 'H809132', 'H90927', 'H809100', 'H80984', 'H90983', 'H809102', 'H909124', 'H9092', 'H80989', 'H90976', 'H909108', 'H90954', 'H909109', 'H90932', 'H80946', 'H90915', 'H80913', 'H9096', 'H90984', 'H80943', 'H909132', 'H90962', 'H80995', 'H909129', 'H809118', 'H809133', 'H90981', 'H809156', 'H909115', 'H90961', 'H90964', 'H80912', 'H90916', 'H90985', 'H90931', 'H909126', 'H80911', 'H90955', 'H909130', 'H90944', 'H90957', 'H80970', 'H90937', 'H90970', 'H90977', 'H90986', 'H809131', 'H909136', 'H80976', 'H80941', 'H80994', 'H90929', 'H90936', 'H809129', 'H809128', 'H80922', 'H909103', 'H909105', 'H809143', 'H8091', 'H90945', 'H80931', 'H909125', 'H80923']

query_fetch_all_anforande = '''MATCH (p:Protokoll {dok_id: $current_dok_id})
            OPTIONAL MATCH (d:Debatt)-[:DOCUMENTED_IN]->(p)
            OPTIONAL MATCH (t:Talare)-[:DELTAR_I]->(d)
            OPTIONAL MATCH (t)-[:HALLER]->(a:Anforande)
            RETURN p.dok_id, a.anforande_id, a.anforande_text
            """, current_dok_id=current_dok_id'''

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1500,   # 1500 characters per chunk
    chunk_overlap=200  # 200 characters overlap to maintain context
)

def test_connection():
    with driver.session() as session:
        result = session.run("RETURN 'Hello, Neo4j!' AS message")
        for record in result:
            print(record["message"])

def generate_chunk_id(anforande_id, chunk_index):
    return hashlib.md5(f"{anforande_id}-{chunk_index}".encode()).hexdigest()

def fetch_all_anforande(current_dok_id=None):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (a:Anforande)
            WHERE NOT (a)-[:HAS_CHUNK]->()
            RETURN a.anforande_id as anforande_id, a.anforande_text as anforande_text
            """, 
        )
        data = []
        for record in result:
            anforande_id = record["anforande_id"]
            anforande_text = record["anforande_text"]
            if anforande_id and anforande_text:
                data.append({"anforande_id": anforande_id, "text": anforande_text})
        return data
    

def store_chunks_in_neo4j(chunks, anforande_id):
    with driver.session() as session:
        for i, chunk in enumerate(chunks):
            chunk_id = generate_chunk_id(anforande_id, i)

            # Check if chunk already exists
            existing = session.run(
                "MATCH (c:Chunk {chunk_id: $chunk_id}) RETURN c LIMIT 1",
                chunk_id=chunk_id
            ).single()

            if existing:
                print(f"Chunk {chunk_id} already exists. Skipping...")
                continue

            # Only generate embedding if chunk doesn't exist
            embedding = embeddings_model.embed_query(chunk)

            # Create the chunk node and relationship
            session.run(
                """
                MATCH (a:Anforande {anforande_id: $anforande_id})
                MERGE (c:Chunk {chunk_id: $chunk_id})
                ON CREATE SET c.chunk_index = $chunk_index, 
                              c.text = $text, 
                              c.embedding = $embedding,
                              c.anforande_id = $anforande_id
                MERGE (a)-[:HAS_CHUNK]->(c)
                """,
                anforande_id=anforande_id,
                chunk_index=i,
                chunk_id=chunk_id,
                text=chunk,
                embedding=embedding
            )


test_connection()

#for id in dok_ids:
#print(f"\n=== Processing dok_id: {id} ===")
data = fetch_all_anforande()

#print(f"Fetched {sum(len(v) for v in data.values())} speeches in total for doc {id}.")

if not data:
    print("No speeches found in the database.")
else:
    print("hellooooooooo")
    print(f"Found {len(data)} total items") #Anforande(s) for Protokoll {dok_id}.")
    for item in data: #.items(): # dok_id, 
        #for item in anforande:
        anforande = item["text"]
        anforande_id = item["anforande_id"]
        chunks = text_splitter.split_text(anforande)
        print(f"Generated {len(chunks)} chunks for Anforande {anforande_id}.")
        store_chunks_in_neo4j(chunks, anforande_id)

    print("All chunks stored successfully.")

driver.close()