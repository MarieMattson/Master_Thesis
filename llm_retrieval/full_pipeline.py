import os
from IPython import embed
from dotenv import load_dotenv
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage
import numpy as np
from colorama import Fore, Style



load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USER")
NEO4J_PASSWORD =  os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
graph = Neo4jGraph()

graph.refresh_schema()
enhanced_graph = Neo4jGraph(enhanced_schema=True)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
chain = GraphCypherQAChain.from_llm(
    graph=enhanced_graph, 
    llm=llm, 
    verbose=True, 
    allow_dangerous_requests=True,
    return_intermediate_steps=True)


system_message = SystemMessage(
    content="""
        You're job is to respond to user queries about a debate from the Swedish parliament (riksdagen).
        The query and data will be in Swedish. 
        When generating a query, it must not be too specific. 
        For example, If asked about what Eva Flyborg said about a certain topic,
        You may retrieve all nodes related to Eva Flyborg.
        Also note that speakers's name are in caps, like "EVA FLYBORG"

        **Requirements:**
        - Only return a valid Cypher query—no explanations or summaries.
        - The speaker's name will be in uppercase with a party label (e.g., "JESSICA POLFJÄRD (M)").
        - The query should find the speaker’s "Anförande" nodes and related "Chunk" nodes.
        - The Protokoll ID will be provided (e.g., "H00998").
        - **Output ONLY the Cypher query.**
        - When generating the query, you must always include the `chunk_id` in the `RETURN` clause, along with the `text` and `anforande_text` of the nodes. 

        Example Cypher query format:
        MATCH (t:Talare {name: "EVA FLYBORG (FP)"}) 
        MATCH (t)-[:HALLER]->(a:Anforande) 
        MATCH (a)-[:HAS_CHUNK]->(c:Chunk)
        MATCH (t)-[:DELTAR_I]->(d:Debatt)-[:DOCUMENTED_IN]->(p:Protokoll {dok_id: "H0091"})
        RETURN a.anforande_text, c.text, c.chunk_id

        The knowledge graph has the following properties:
        Node properties:
        - **Chunk**
        - `text`: STRING Example: "Det var dagens sista fråga. Jag vill tacka de närv"
        - `chunk_index`: INTEGER Min: 0, Max: 0
        - `chunk_id`: STRING Example: "d15dc718-661c-4d1b-9d5c-a3e4370e1713"
        - **Protokoll**
        - `dok_hangar_id`: STRING Example: "2804730"
        - `dok_rm`: STRING Available options: ['2012/13']
        - `dok_datum`: STRING Example: "2012-09-18 00:00:00"
        - `dok_id`: STRING Example: "H0091"
        - `dok_titel`: STRING Example: "Riksdagens protokoll 2012/13:1 Tisdagen den 18 sep"
        - **Debatt**
        - `avsnittsrubrik`: STRING Example: "Parentation"
        - `kammaraktivitet`: STRING Example: "Unknown"
        - **Talare**
        - `name`: STRING Example: "ANDERS BORG"
        - `party`: STRING Available options: ['Unknown', 'S', 'MP', 'M', 'FP', 'C', 'KD', 'SD', 'V']
        - **Anforande**
        - `replik`: STRING Example: "N"
        - `anforande_nummer`: STRING Example: "1"
        - `intressent_id`: STRING Example: "0000000000000"
        - `underrubrik`: STRING Example: "Unknown"
        - `rel_dok_id`: STRING Example: "Unknown"
        - `anforande_text`: STRING Example: "  Ärade riksdagsledamöter! Jag vill hälsa er välko"
        - `anforande_id`: STRING Example: "B46BE744-50D8-469A-A297-AC6EB328EBEB"
        Relationship properties:
        The relationships:
        (:Debatt)-[:DOCUMENTED_IN]->(:Protokoll)
        (:Talare)-[:DELTAR_I]->(:Debatt)
        (:Talare)-[:HALLER]->(:Anforande)
        (:Anforande)-[:HAS_CHUNK]->(:Chunk) 
    """
)


def generate_cypher_query(user_query):
    """Uses LangChain's QA Chain to generate a Cypher query to get relevant nodes."""
    response = chain.invoke({"query": user_query, "messages": [system_message, HumanMessage(content=user_query)]})
    cypher_query = response["intermediate_steps"][0]["query"]
    return cypher_query

def get_relevant_nodes(cypher_query):
    """Runs the Cypher query in Neo4j and retrieves relevant nodes (chunks of text)."""
    result = graph.query(cypher_query)
    return result  # This contains the chunks from Jessica Polfjärd's speeches

def perform_semantic_search(query_text, relevant_nodes, top_k=6):
    """Performs vector search only on the retrieved nodes."""

    # Generate embedding for the query (Fixed this line)
    embedding = embeddings.embed_query(query_text)  # Directly returns a list


    # Prepare embedding search within the filtered nodes
    chunk_ids = [node.get("chunk_id") or node.get("c.chunk_id") for node in relevant_nodes if node.get("chunk_id") or node.get("c.chunk_id")]

    # Query Neo4j for similar chunks among filtered results
    result = graph.query("""
    MATCH (c:Chunk)
    WHERE c.chunk_id IN $chunk_ids  // Restrict to retrieved chunks
    WITH c, gds.alpha.similarity.cosine(c.embedding, $embedding) AS score
    RETURN c.text AS text, c.chunk_id AS chunk_id, score
    ORDER BY score DESC
    LIMIT $top_k
    """, {"embedding": embedding, "top_k": top_k, "chunk_ids": chunk_ids})


    return result

if __name__ == "__main__":
    query = (
        """
        Hur argumenterar JESSICA POLFJÄRD för sänkt restaurangmoms och fler jobb 
        i debatten från Protokoll H00998? Du MÅSTE returnera a.anforande_text, c.text, c.chunk_id, chunk.embedding!
        Generera en Cypher query som begränsar till den specifika debatten 
        och den aktuella talarens anförande, men undvik för många filter.
        """)   
    
     # Step 1: Generate Cypher query to get Jessica Polfjärd's statements
    cypher_query = generate_cypher_query(query)
    print("🔍 Generated Cypher Query:\n", cypher_query)

    # Step 2: Run Cypher query to get relevant nodes
    relevant_nodes = get_relevant_nodes(cypher_query)
    print("📌 Retrieved relevant nodes:", len(relevant_nodes))

    # Step 3: Perform semantic search within retrieved nodes
    refined_results = perform_semantic_search(query, relevant_nodes)
    print("🔎 Top semantic matches:")
    for chunk in refined_results:
        print(f"{chunk['text']} (Score: {chunk['score']})")

    # Final response combining both sources
    final_response = {
        "structured_query_response": relevant_nodes,  # Filtered by Cypher
        "semantic_matches": refined_results  # Ranked by vector search
    }

    print("📢 Final Response:", final_response)
    
    #response = chain.invoke({"query": query, "messages": [system_message, HumanMessage(content=query)]})

    #print(response)
    '''print("="*30, "Semantic search", "="*30)

    # Step 1: Convert the user query into an embedding
    embedded_query = embeddings.embed_query(query)


    # Step 2: Perform vector search in Neo4j to find relevant nodes
    result = graph.query("""
    CALL db.index.vector.queryNodes('chunkVector', 6, $embedding)
    YIELD node, score
    RETURN node.text, score
    """, {"embedding": embedded_query})

    # Step 3: Process the search results
    relevant_texts = []
    for row in result:
        relevant_texts.append(row['node.text'])
        print(f"Node Text: {row['node.text']} | Score: {row['score']}")

    # Step 4: Pass the relevant texts and query to LangChain for detailed response
    # Now, LangChain will take the result of the vector search and generate a detailed response.
    response = chain.invoke({
        "query": query, 
        "messages": [
            system_message, 
            HumanMessage(content=query), 
            HumanMessage(content=" ".join(relevant_texts))  # Passing the relevant chunk texts as context
        ]
    })

    relevant_texts = []
    nodes_with_scores = []
    
    for row in result:
        nodes_with_scores.append((row['node.text'], row['score']))

    # Sort by score in descending order (highest score first)
    nodes_with_scores.sort(key=lambda x: x[1], reverse=True)

    # Display the nodes with their scores
    print("="*30, "Top Scoring Nodes", "="*30)
    for idx, (text, score) in enumerate(nodes_with_scores[:5]):  # Show top 5 results
        print(f"{Fore.MAGENTA}Node {idx+1}: {Style.BRIGHT}{text}")
        print(f"{Fore.CYAN}Score: {score}{Style.RESET_ALL}")
        print("="*30)

    # Step 4: Pass the relevant texts and query to LangChain for detailed response
    relevant_texts = [text for text, _ in nodes_with_scores[:5]]  # Taking top 5 nodes as context
    #response = chain.invoke({
    #    "query": query, 
    #    "messages": [
    #        system_message, 
    #        HumanMessage(content=query), 
    #        HumanMessage(content=" ".join(relevant_texts))  # Passing the relevant chunk texts as context
    #    ]
    #})

    # Print the final response with visual separation and color
    #print("="*30, "Response from LangChain", "="*30)
    #print(Fore.GREEN + Style.BRIGHT + "="*50)
    #print(Fore.YELLOW + "Generated Response:")
    #print(Fore.CYAN + Style.BRIGHT + response['result'])
    #print(Fore.GREEN + "="*50 + Style.RESET_ALL)'''