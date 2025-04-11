import json
import traceback
from llm_retrieval.full_pipeline_class import GraphRAG

graph_rag = GraphRAG()
with open("/mnt/c/Users/User/thesis/data_import/data_small_size/inital_test_with_big_data.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)


for entry in dataset:
    try:
        question = entry["qa_pair"]["question"]
        
        print(f"\nProcessing question: {question}")
        
        user_query = question

        cypher_query = graph_rag.translate_to_cypher(user_query)
        print("Generated Cypher Query:", cypher_query)

        retrieved_nodes = graph_rag.retrieve_nodes(cypher_query)
        print(f"Retrieved {len(retrieved_nodes)} nodes.")

        #cosine_ranked_nodes = graph_rag.rank_nodes_by_similarity(user_query, retrieved_nodes)
        bm25_ranked_nodes = graph_rag.rank_nodes_with_BM25(user_query, retrieved_nodes)
        #cosine_final_response = graph_rag.generate_response(cosine_ranked_nodes, user_query)
        #print("Final Response:", cosine_final_response)
        bm25_final_response = graph_rag.generate_response(bm25_ranked_nodes, user_query)

        #entry["graph_RAG"]["answer"] = final_response.strip()
        #entry["graph_RAG"]["context"] = [node["node"]["a.anforande_id"] for node in cosine_ranked_nodes]
        #entry["graph_RAG"]["cypher_query"] = cypher_query

        print(bm25_final_response)
        entry["graph_RAG_BM25"]["answer"] = bm25_final_response.strip()
        entry["graph_RAG_BM25"]["context"] = [node["node"]["a.anforande_id"] for node in bm25_ranked_nodes]
        entry["graph_RAG_BM25"]["cypher_query"] = cypher_query
    

    except Exception as e:
        print(f"⚠️ Error processing entry: {question}")
        print(traceback.format_exc())

with open("/mnt/c/Users/User/thesis/data_import/data_small_size/inital_test_with_big_data.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print("\nProcessing complete!")
