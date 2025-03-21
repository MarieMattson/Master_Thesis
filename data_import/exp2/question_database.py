import json
from llm_retrieval.full_pipeline_class import GraphRAG

# Initialize RAG
graph_rag = GraphRAG()

# Load your dataset (assuming it's stored in a JSON file)
with open("/mnt/c/Users/User/thesis/data_import/exp2/mini_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Store results
rag_annotated_data = []

# Iterate through each entry in the dataset
for entry in dataset:
    question = entry["qa_pair"]["question"]
    
    print(f"\nProcessing question: {question}")
    
    # Construct user query
    user_query = f"""
    {question}
    Du MÅSTE returnera c.anforande_id, a.anforande_text, c.text, c.embedding!
    Generera en Cypher query som begränsar till den specifika debatten 
    och den aktuella talarens anförande, men undvik för många filter.
    Om en fråga rör ett partis åsikt, bör du returnera noder från politiker från partiet.
    Exemplvis MATCH (t:Talare{{party:"MP"}}) RETURN t
    """

    # Generate Cypher Query
    cypher_query = graph_rag.translate_to_cypher(user_query)
    print("Generated Cypher Query:", cypher_query)

    # Retrieve nodes
    retrieved_nodes = graph_rag.retrieve_nodes(cypher_query)
    print(f"Retrieved {len(retrieved_nodes)} nodes.")

    # Rank nodes by similarity
    ranked_nodes = graph_rag.rank_nodes_by_similarity(user_query, retrieved_nodes)

    # Generate final response
    final_response = graph_rag.generate_response(ranked_nodes, user_query)
    print("Final Response:", final_response)

    rag_annotator_output = {
        "question": question,
        "answer": final_response.strip(), 
        "context": [f"{item['node']['c.anforande_id']}" for i, item in enumerate(ranked_nodes, start=1)]

    }



    entry["rag_annotator"] = rag_annotator_output  
    rag_annotated_data.append(entry)

with open("rag_output.json", "w", encoding="utf-8") as f:
    json.dump(rag_annotated_data, f, ensure_ascii=False, indent=4)

print("\nProcessing complete! Results saved to rag_output.json.")
