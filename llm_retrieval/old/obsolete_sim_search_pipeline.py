"""
Succesfully performs similarity search, returns nodes and their scores.
However, when i try to generate a response, I keep reaching token limit by A LOT (90 000).
Something is clearly wrong, but idk what?
"""
import os
from dotenv import load_dotenv
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage
import langchain_community.vectorstores
from openai import OpenAI

# Load environment variables
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Neo4j Graph
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
graph.refresh_schema()
enhanced_graph = Neo4jGraph(enhanced_schema=True)

# Initialize OpenAI client
llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Define system message for GraphCypherQAChain
system_message = SystemMessage(
    content="""
        Your job is to respond to queries about Swedish parliamentary debates.
        The data is in Swedish, and queries should not be overly specific.
        Ensure the responses are relevant based on both structured Cypher queries and vector search results.
    """
)

chain = GraphCypherQAChain.from_llm(
    graph=enhanced_graph, llm=llm, verbose=False, allow_dangerous_requests=True,
    return_intermediate_steps=True
)

def get_semantic_matches(query_text, top_k=6):
    """Retrieve semantically similar text chunks using vector search in Neo4j."""
    
    # Generate embedding for the query
    query_embedding = embedding.embed_query(query_text)
    # Query Neo4j for similar chunks
    result = graph.query("""
    CALL db.index.vector.queryNodes('chunkVector', $top_k, $embedding)
    YIELD node, score
    RETURN node.text AS text, node.chunk_id AS chunk_id, score
    """, {"embedding": query_embedding, "top_k": top_k})

    return result



def get_filtered_semantic_matches(query_text, top_k=6):
    """Retrieve semantically similar text chunks using vector search in Neo4j with filtering."""
    
    # Generate embedding for the query
    embedding = embedding.embed_query(query_text)
    
    # Construct a filter based on the query or other parameters
    metadata_filter, _ = langchain_community.vectorstores.falkordb_vector.construct_metadata_filter(filter={"talare": "JESSICA POLFJÄRD", "debate_id": "H00998"})
    
    # Query Neo4j with a filter for specific speaker and debate
    result = graph.query("""
    CALL db.index.vector.queryNodes('chunkVector', $top_k, $embedding)
    YIELD node, score
    WHERE node.metadata CONTAINS $metadata_filter
    RETURN node.text AS text, node.chunk_id AS chunk_id, score
    """, {"embedding": embedding, "top_k": top_k, "metadata_filter": metadata_filter})

    return result 



if __name__ == "__main__":
    query = (
        """
        Hur argumenterar JESSICA POLFJÄRD för sänkt restaurangmoms och fler jobb 
        i debatten från Protokoll H00998? Du MÅSTE returnera a.anforande_text, c.text, c.chunk_id, chunk.embedding!
        Generera en Cypher query som begränsar till den specifika debatten 
        och den aktuella talarens anförande, men undvik för många filter.
        """) 
    # Step 1: Retrieve semantically similar chunks
    similar_chunks = get_semantic_matches(query)
    print("🔍 Top semantic matches:")
    for chunk in similar_chunks:
        print(f"{chunk['text']} (Score: {chunk['score']})")

    # Step 2: Run structured Cypher QA
    response = chain.invoke({"query": query, "messages": [system_message, HumanMessage(content=query)]})

    # Step 3: Combine both sources
    final_response = {
        "structured_query_response": response,
        "semantic_matches": similar_chunks
    }

    print("📌 Final Response:", final_response)
