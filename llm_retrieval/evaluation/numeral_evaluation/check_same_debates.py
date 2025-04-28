import json

output_path = "/mnt/c/Users/User/thesis/data_import/data_small_size/data/evaluated_dataset.json"
original_data = "/mnt/c/Users/User/thesis/data_import/data_small_size/data/dataset_small.json"
retreival_models = ["graph_RAG_cosine","graph_RAG_bm25", "cosine_RAG"]

with open(output_path, "r") as f:
    output_data = json.load(f)
with open(original_data, "r") as f:
    original_data = json.load(f) 

# Build a lookup dictionary for faster access
original_lookup = {d["anforande_id"]: d for d in original_data}


for i, d in enumerate(output_data):
    reference_dok_id = d["dok_id"]
    reference_debate_title = d["avsnittsrubrik"]

    for model in retreival_models:
        model_relevance = []

        for i, retrieved_speech in enumerate(d[model]["context"]):
            if i >= 6:
                break
            relevance = False

            # Look up the retrieved speech
            original = original_lookup.get(retrieved_speech)
            if original:
                original_retrieved_dok_id = original["dok_id"]
                original_debate_title = original["avsnittsrubrik"]

                # Now compare
                if reference_dok_id == original_retrieved_dok_id:
                    if reference_debate_title == original_debate_title:
                        relevance = True

                
            # Add the relevance (True/False) for this speech in the model's list
            model_relevance.append(relevance)
            if "relevance" not in d:
                d["relevance"] = {}
            d["relevance"][model] = model_relevance

output_path = "/mnt/c/Users/User/thesis/data_import/data_small_size/data/evaluated_dataset_with_relevance.json"

#with open(output_path, "w", encoding="utf-8") as f:
#    json.dump(output_data, f, indent=2, ensure_ascii=False)

#print(f"Relevance information added and saved to {output_path}")


with open(output_path, "r") as f:
    output_data = json.load(f)

relevance_counts = {
    "graph_RAG_cosine": {True: 0, False: 0},
    "graph_RAG_bm25": {True: 0, False: 0},
    "cosine_RAG": {True: 0, False: 0}
}

for d in output_data:
    relevance = d.get("relevance", {})  # Get the relevance data for the current document
    for model in relevance:
        # Count True and False values for each model
        true_count = relevance[model].count(True)
        false_count = relevance[model].count(False)

        # Update the count in the dictionary
        relevance_counts[model][True] += true_count
        relevance_counts[model][False] += false_count

# Print out the counts for each model
print("Relevance Counts:")
for model in relevance_counts:
    print(f"{model}: True: {relevance_counts[model][True]}, False: {relevance_counts[model][False]}")


# Initialize variables to store total relevant and retrieved documents for each model
precision_results = {}
recall_results = {}
map_results = {}

# Iterate through the data to calculate Precision, Recall, and MAP for each model
for model in ["graph_RAG_cosine", "graph_RAG_bm25", "cosine_RAG"]:
    total_relevant = 0
    total_retrieved = 0
    total_relevant_retrieved = 0
    average_precision = 0
    query_count = 0

    # Loop over each query (document) in the dataset
    for d in output_data:
        relevance = d.get("relevance", {}).get(model, [])
        retrieved_count = len(relevance)
        relevant_count = sum(relevance)  # Count of True values (relevant documents)

        if relevant_count > 0:
            query_count += 1
            total_relevant += relevant_count
            total_retrieved += retrieved_count
            total_relevant_retrieved += relevant_count
            
            # Precision = (Relevant Retrieved) / (Total Retrieved)
            precision = relevant_count / retrieved_count if retrieved_count > 0 else 0

            # Recall = (Relevant Retrieved) / (Total Relevant)
            recall = relevant_count / relevant_count if relevant_count > 0 else 0

            # Average Precision (AP) for the current query
            ap = 0
            relevant_retrieved = 0
            for idx, rel in enumerate(relevance):
                if rel:
                    relevant_retrieved += 1
                    ap += relevant_retrieved / (idx + 1)
            ap /= relevant_count if relevant_count > 0 else 1

            average_precision += ap

    # Store the results for this model
    precision_results[model] = total_relevant_retrieved / total_retrieved if total_retrieved > 0 else 0
    recall_results[model] = total_relevant_retrieved / total_relevant if total_relevant > 0 else 0
    map_results[model] = average_precision / query_count if query_count > 0 else 0

# Output the results
print("Precision Results:")
for model, precision in precision_results.items():
    print(f"{model}: Precision = {precision:.4f}")

print("\nRecall Results:")
for model, recall in recall_results.items():
    print(f"{model}: Recall = {recall:.4f}")

print("\nMAP Results:")
for model, map_score in map_results.items():
    print(f"{model}: MAP = {map_score:.4f}")