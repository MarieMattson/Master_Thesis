"""THIS IS A MESS! DON'T LOOK AT IT!"""
import os
from IPython import embed
from dotenv import load_dotenv
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage
import numpy as np
from colorama import Fore, Style
from sklearn.metrics.pairwise import cosine_similarity


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
    return_intermediate_steps=True,
    validate_cypher=True)



system_message = SystemMessage(
    content="""
        You're job is to respond to user queries about a debate from the Swedish parliament (riksdagen).
        The query and data will be in Swedish. 
        The cypher query should consist of two parts:
        First you should filter out everything that is relevant for the speaker,
        for example, If asked about what Eva Flyborg said about a certain topic,
        You may retrieve all nodes related to Eva Flyborg.
        Second, you shold filter out the nodes that have the highest similarity score to the user query        
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
    """
)
'''system_message = SystemMessage(
    content = """
            You are a useful cypher query assistant     
            Your task is to generate a valid cypher query based on the user input
            Only return the Cypher query. Do not run the query!
            
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
        """)'''

def generate_cypher_query(user_query):
    """Uses LangChain's QA Chain to generate a Cypher query to get relevant nodes."""
    print("User query:", user_query)
    
    # Ensure chain.invoke() returns a response (even if it's an error message)
    response = chain.invoke({"query": user_query}) 
    
    # Print the full response to see the structure
    print("Response:", response)
    
    # Check if the expected 'intermediate_steps' exists in the response
    if "intermediate_steps" in response and len(response["intermediate_steps"]) > 0:
        cypher_query = response["intermediate_steps"][0]["query"]
        print("Cypher query:", cypher_query)
        return cypher_query
    else:
        print("Error: No intermediate steps found in response")
        return None
    
'''def generate_cypher_query(user_query):
    """Uses LangChain's QA Chain to generate a Cypher query to get relevant nodes."""
    print(user_query)
    response = chain.invoke({"query": user_query, "messages": [system_message, HumanMessage(content=user_query)]})

    print(response)
    cypher_query = response["intermediate_steps"][0]["query"]
    print(cypher_query)
    return cypher_query

def query_graph(cypher_query):
    """Runs the Cypher query in Neo4j and retrieves relevant nodes (chunks of text)."""
    result = graph.query(cypher_query)
    return result'''

def perform_semantic_search(query_text, relevant_nodes, top_k=6):
    """
    output:
    [{'a.anforande_text': '\r\nHerr talman! Jag träffade i förra veckan.....', 
    'c.chunk_id': '71799eeb-0b67-41d9-8a62-5ee46656b4df', 
    'c.embedding': [0.004316803999245167, 0.004593313671648502,.....]}]
    """

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
        i debatten från Protokoll H00998? Du MÅSTE returnera a.anforande_text, c.text, c.chunk_id!
        Generera en Cypher query som begränsar till den specifika debatten 
        och den aktuella talarens anförande, men undvik för många filter.
        """)   
    
     # Step 1: Generate Cypher query to get Jessica Polfjärd's statements
    cypher_query = generate_cypher_query(query)
    print("🔍 Generated Cypher Query:\n", cypher_query)

    # Step 2: Run Cypher query to get relevant nodes
    relevant_nodes = query_graph(cypher_query)
    print("📌 Retrieved relevant nodes:", len(relevant_nodes))

    # Step 3: Filter nodes to get the embeddings
    #relevant_nodes = filter_nodes(relevant_nodes)

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
    print("="*30, "Semantic search", "="*30)

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
    response = chain.invoke({
        "query": query, 
        "messages": [
            system_message, 
            HumanMessage(content=query), 
            HumanMessage(content=" ".join(relevant_texts))  # Passing the relevant chunk texts as context
        ]
    })

    # Print the final response with visual separation and color
    print("="*30, "Response from LangChain", "="*30)
    print(Fore.GREEN + Style.BRIGHT + "="*50)
    print(Fore.YELLOW + "Generated Response:")
    print(Fore.CYAN + Style.BRIGHT + response['result'])
    print(Fore.GREEN + "="*50 + Style.RESET_ALL)