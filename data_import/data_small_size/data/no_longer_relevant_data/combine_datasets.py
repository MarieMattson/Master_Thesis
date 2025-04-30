import json

# Paths to the data files
graph_path = "/mnt/c/Users/User/thesis/data_import/data_small_size/data/result_faiss_qa_dataset_part4.json"
cosine_path = "/mnt/c/Users/User/thesis/data_import/data_small_size/data/result_qa_dataset_part4.json"
output_path = "/mnt/c/Users/User/thesis/data_import/data_small_size/data/final_combined_result_part4.json"

# Load graph_data and cosine_data
with open(graph_path, "r", encoding="utf-8") as f: 
    graph_data = json.load(f)

with open(cosine_path, "r", encoding="utf-8") as f:
    cosine_data = json.load(f)

# Iterate through cosine_data and graph_data simultaneously
for entry in cosine_data:
    question = entry["qa_pair"]["question"]

    # Find the corresponding entry in graph_data by matching the question
    for graph_entry in graph_data:
        if graph_entry["qa_pair"]["question"] == question:

            
            # Update the fields in cosine_data's 'cosine_RAG' with non-empty fields from graph_data
            if not entry["graph_RAG_cosine"]["answer"]:
                entry["graph_RAG_cosine"]["answer"] = graph_entry["graph_RAG_cosine"]["answer"]
            if not entry["graph_RAG_cosine"]["context"]:
                entry["graph_RAG_cosine"]["context"] = graph_entry["graph_RAG_cosine"]["context"]
            if not entry["graph_RAG_cosine"]["cypher_query"]:
                entry["graph_RAG_cosine"]["cypher_query"] = graph_entry["graph_RAG_cosine"]["cypher_query"]
            if entry["graph_RAG_cosine"]["number_of_nodes"] == 0:
                try:
                    entry["graph_RAG_cosine"]["number_of_nodes"] = graph_entry["graph_RAG_cosine"]["number_of_nodes"]
                except KeyError:
                    pass
            if not entry["graph_RAG_bm25"]["answer"]:
                entry["graph_RAG_bm25"]["answer"] = graph_entry["graph_RAG_bm25"]["answer"]
            if not entry["graph_RAG_bm25"]["context"]:
                entry["graph_RAG_bm25"]["context"] = graph_entry["graph_RAG_bm25"]["context"]
            if not entry["graph_RAG_bm25"]["cypher_query"]:
                entry["graph_RAG_bm25"]["cypher_query"] = graph_entry["graph_RAG_bm25"]["cypher_query"]
            if entry["graph_RAG_bm25"]["number_of_nodes"] == 0:
                try:
                    entry["graph_RAG_bm25"]["number_of_nodes"] = graph_entry["graph_RAG_bm25"]["number_of_nodes"]
                except KeyError:
                    pass
            # If there are additional fields you want to update, do it similarly here
            break  # No need to continue once a matching question is found

# Save the combined data to a new file
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(cosine_data, f, ensure_ascii=False, indent=4)

print("\nData combined and saved successfully!")
