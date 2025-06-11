import json
import traceback
from llm_retrieval.KGRAG_full_pipeline import GraphRAG
# Change paths to large dataset if needed
#/mnt/c/Users/User/thesis/data_import/data_small_size/data/qa_dataset_part1.json
#/mnt/c/Users/User/thesis/data_import/data_small_size/data/qa_dataset_part2.json
#/mnt/c/Users/User/thesis/data_import/data_small_size/data/qa_dataset_part3.json
#/mnt/c/Users/User/thesis/data_import/data_small_size/data/qa_dataset_part3.json
#/mnt/c/Users/User/thesis/data_import/data_small_size/data/null_queries.json
graph_rag = GraphRAG()
with open("/mnt/c/Users/User/thesis/data_import/data_small_size/data/null_queries.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

for idx, entry in enumerate(dataset):
    try:
        question = entry["qa_pair"]["question"]
        
        print(f"\nProcessing question: {question}")
        
        try:
            cypher_query = graph_rag.translate_to_cypher(question)
            print("Generated Cypher Query:", cypher_query)
            entry["graph_RAG_cosine"]["cypher_query"] = cypher_query
            entry["graph_RAG_bm25"]["cypher_query"] = cypher_query
        except Exception as e:
            entry["graph_RAG_cosine"]["cypher_query"] = f"Cypher query failed: {e}"
            entry["graph_RAG_bm25"]["cypher_query"] = f"Cypher query failed: {e}"
            print(f"Cypher translation failed: {e}")
            continue


        try:
            retrieved_nodes = graph_rag.retrieve_nodes(cypher_query)
            number_of_nodes = len(retrieved_nodes)
            entry["graph_RAG_cosine"]["number_of_nodes"] = number_of_nodes
            entry["graph_RAG_bm25"]["number_of_nodes"] = number_of_nodes
            print(f"Retrieved {number_of_nodes} nodes.")
        except Exception as e:
            entry["graph_RAG_cosine"]["number_of_nodes"] = f"Node retrieval failed: {e}"
            entry["graph_RAG_bm25"]["number_of_nodes"] = f"Node retrieval failed: {e}"
            print(f"Node retrieval failed: {e}")
            continue

        try:
            cosine_ranked_nodes = graph_rag.rank_nodes_by_similarity(question, retrieved_nodes)
            entry["graph_RAG_cosine"]["context"] = [node["node"]["a.anforande_id"] for node in cosine_ranked_nodes]
            print("Ranked nodes by cosine similarity.")
        except Exception as e:
            entry["graph_RAG_cosine"]["context"] = f"Cosine ranking failed: {e}"
            print(f"Ranking failed: {e}")      
            continue
        
        try:
            bm25_ranked_nodes = graph_rag.rank_nodes_with_BM25(question, retrieved_nodes)
            entry["graph_RAG_bm25"]["context"] = [node["node"]["a.anforande_id"] for node in bm25_ranked_nodes]
            print("Ranked nodes by BM25.")
        except Exception as e:
            entry["graph_RAG_bm25"]["context"] = f"BM25 ranking failed: {e}"
            print(f"BM25 ranking failed: {e}")
            continue
        

        try:
            print("Generating final response...")
            cosine_final_response = graph_rag.generate_response(cosine_ranked_nodes, question)
            print("Final Response Cosine:", cosine_final_response)
            entry["graph_RAG_cosine"]["answer"] = cosine_final_response
        except Exception as e:
            entry["graph_RAG_cosine"]["answer"] = f"Cosine response generation failed: {e}"
            print(f"Response generation failed: {e}")
            continue
        
        try:
            print("Generating final response for BM25...")
            bm25_final_response = graph_rag.generate_response(bm25_ranked_nodes, question)
            print("Final Response BM25:", bm25_final_response)
            entry["graph_RAG_bm25"]["answer"] = bm25_final_response
        except Exception as e:
            entry["graph_RAG_bm25"]["answer"] = f"BM25 response generation failed: {e}"
            print(f"Response generation failed: {e}")
            continue

    except Exception as e:
        print(f"⚠️ Error processing entry: {question}")
        print(traceback.format_exc())


with open("/mnt/c/Users/User/thesis/data_import/data_small_size/data/result_qa_dataset_null.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print("\nProcessing complete!")
