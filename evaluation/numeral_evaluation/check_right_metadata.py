"""Here I check if the metadata from the retrieved context is the same as the metadata from the query.
What metadata is relevevant will be based on the question type."""

import json
from sklearn.metrics import average_precision_score

question_types_person = ["generate_qa_inference_person", "generate_qa_comparison_person"]
question_types_temporal = ["generate_qa_temporal"]
question_types_party = ["generate_qa_inference_party", "generate_qa_comparison_party"] 

# small dataset
#experiment_result = "/mnt/c/Users/User/thesis/data_import/data_small_size/data/final_248_combined_result.json"
#original_data = "/mnt/c/Users/User/thesis/data_import/data_small_size/data/dataset_small.json"

#large dataset
experiment_result = "/mnt/c/Users/User/thesis/data_import/data_large_size/data/full_result_merged.json"
original_data = "/mnt/c/Users/User/thesis/data_import/data_large_size/filtered_riksdag.json"

retreival_models = ["graph_RAG_cosine", "graph_RAG_bm25", "cosine_RAG"]

with open(experiment_result, "r") as f:
    output_data = json.load(f)
with open(original_data, "r") as f:
    original_data = json.load(f)

original_lookup = {d["anforande_id"]: d for d in original_data}

for i, d in enumerate(output_data):
    if d["qa_type"] in question_types_person:
        relevant_metadata = "talare"
        goal_metadata = d[relevant_metadata]
    elif d["qa_type"] in question_types_temporal:
        relevant_metadata = "dok_datum"
        goal_metadata = d[relevant_metadata][0:7]
        print(f"goal_metadata: {goal_metadata}")
    elif d["qa_type"] in question_types_party:
        relevant_metadata = "parti"
        goal_metadata = d[relevant_metadata]

    
    for model in retreival_models:
        model_relevance = []
        for i, retrieved_speech in enumerate(d[model]["context"]):
            if i >= 6:
                break
            relevance = False
            original = original_lookup.get(retrieved_speech)
            if original:
                # Get the relevant metadata from the original retrieved speech
                original_retrieved_metadata = original.get(relevant_metadata, None)
                
                if original_retrieved_metadata:

                    if relevant_metadata == "dok_datum":
                        original_retrieved_metadata = original_retrieved_metadata[0:7]
                        print(f"original_retrieved_metadata: {original_retrieved_metadata}")

                    # Check if the goal metadata matches the original metadata
                    if goal_metadata == original_retrieved_metadata:
                        relevance = True

            model_relevance.append(relevance)
            if "metadata_relevance" not in d:
                d["metadata_relevance"] = {}
            d["metadata_relevance"][model] = model_relevance



output_path = "/mnt/c/Users/User/thesis/data_import/data_small_size/data/final_248_combined_result_with_relevence.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"Relevance information added and saved to {output_path}")



with open(output_path, "r") as f:
    output_data = json.load(f)

relevance_counts = {
    "graph_RAG_cosine": {True: 0, False: 0},
    "graph_RAG_bm25": {True: 0, False: 0},
    "cosine_RAG": {True: 0, False: 0}
}

mean_relevance_counts = {
    "graph_RAG_cosine": 0,
    "graph_RAG_bm25": 0,
    "cosine_RAG": 0
}

total_combined_relevant = 0
total_combined_documents = 0

for d in output_data:
    relevance = d.get("metadata_relevance", {})
    for model in relevance:
        model_relevance_list = relevance[model]
        true_count = model_relevance_list.count(True)
        false_count = model_relevance_list.count(False)
        total = true_count + false_count

        relevance_counts[model][True] += true_count
        relevance_counts[model][False] += false_count
        mean_relevance_counts[model] += true_count/(true_count + false_count) if (true_count + false_count) > 0 else 0
        

        if total > 0:
            mean_rel = true_count / total

            total_combined_relevant += true_count
            total_combined_documents += total



# Initialize variables to store total relevant and retrieved documents for each model
precision_results = {}
recall_results = {}
map_results = {}
bare_minimum_results = {}
gets_a_pass_results = {}
map_results = {}
zero_count_results = {}
one_count_results = {}
two_count_results = {}
three_count_results = {}
four_count_results = {}
five_count_results = {}
six_count_results = {}

# Iterate through the data to calculate Precision, Recall, and MAP for each model
for model in ["graph_RAG_cosine", "graph_RAG_bm25", "cosine_RAG"]:
    total_relevant = 0
    total_retrieved = 0
    total_relevant_retrieved = 0
    average_precision = 0
    query_count = 0
    bare_minimum_relevant = 0
    gets_a_pass = 0 # meaning has 3 or more relevant documents
    ap_scores = []
    one_count = 0
    zero_count = 0
    two_count = 0
    three_count = 0
    four_count = 0
    five_count = 0
    six_count = 0

    # Loop over each query (document) in the dataset
    for d in output_data:
        relevance = d.get("metadata_relevance", {}).get(model, [])
        retrieved_count = len(relevance)
        relevant_count = sum(relevance)  # Count of True values (relevant documents)
        
        if relevant_count == 0:
            zero_count += 1
        elif relevant_count == 1:
            one_count += 1
        elif relevant_count == 2:
            two_count += 1
        elif relevant_count == 3:
            three_count += 1
        elif relevant_count == 4:
            four_count += 1
        elif relevant_count == 5:
            five_count += 1
        elif relevant_count == 6:
            six_count += 1

        if relevant_count >= 3:
            gets_a_pass += 1

        if relevant_count > 0:
            query_count += 1
            total_relevant += relevant_count
            total_retrieved += retrieved_count
            total_relevant_retrieved += relevant_count

            scores = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]  # Example scores for the retrieved documents

            # Compute average precision
            ap = average_precision_score(relevance, scores)
            ap_scores.append(ap)


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
        else:
            bare_minimum_relevant += 1


    # Store the results for this model
    precision_results[model] = total_relevant_retrieved / total_retrieved if total_retrieved > 0 else 0
    recall_results[model] = total_relevant_retrieved / total_relevant if total_relevant > 0 else 0
    map_results[model] = sum(ap_scores) / len(ap_scores) if ap_scores else 0.0
    bare_minimum_results[model] = bare_minimum_relevant
    gets_a_pass_results[model] = gets_a_pass
    mean_relevance_counts[model] = mean_relevance_counts[model] / len(output_data) if len(output_data) > 0 else 0
    one_count_results[model] = one_count
    zero_count_results[model] = zero_count
    two_count_results[model] = two_count
    three_count_results[model] = three_count
    four_count_results[model] = four_count
    five_count_results[model] = five_count
    six_count_results[model] = six_count
print()

# Print out the counts for each model
print("Relevance Counts:")
for model in relevance_counts:
    print(f"{model}: True: {relevance_counts[model][True]}, False: {relevance_counts[model][False]}")

# Output the results
print("\nPrecision Results:")
for model, precision in precision_results.items():
    print(f"{model}: Precision = {precision:.4f}")

print("\nRecall Results:")
for model, recall in recall_results.items():
    print(f"{model}: Recall = {recall:.4f}")

print("\nMAP Results:")
for model, map_score in map_results.items():
    print(f"{model}: MAP = {map_score:.4f}")

print("\nBare minimum Results:")
for model, mb_score in bare_minimum_results.items():
    print(f"{model}: Bare minimum score = {mb_score}")

print("\nGets a pass Results:")
for model, pass_score in gets_a_pass_results.items():
    print(f"{model}: Gets a pass score = {pass_score}")

print("\nMean Relevance per Model:")
for model, mean_rel in mean_relevance_counts.items():
    print(f"{model}: Mean Relevance = {mean_rel:.4f}")
 
print("\nCounts of Relevant Documents:")
print(f"Zero Count: {zero_count_results}")
print(f"One Count: {one_count_results}")
print(f"Two Count: {two_count_results}")
print(f"Three Count: {three_count_results}") 
print(f"Four Count: {four_count_results}")
print(f"Five Count: {five_count_results}")
print(f"Six Count: {six_count_results}")


# ---- GROUPING BY METADATA TYPE ----
grouped_by_metadata_type = {
    "person": [],
    "party": [],
    "temporal": []
}

for d in output_data:
    qa_type = d.get("qa_type")
    if qa_type in question_types_person:
        grouped_by_metadata_type["person"].append(d)
    elif qa_type in question_types_party:
        grouped_by_metadata_type["party"].append(d)
    elif qa_type in question_types_temporal:
        grouped_by_metadata_type["temporal"].append(d)


# ---- GROUPING BY QUESTION NATURE ----
grouped_by_question_nature = {
    "inference": [],
    "comparison": [],
    "temporal": []
}

for d in output_data:
    qa_type = d.get("qa_type")
    if "inference" in qa_type:
        grouped_by_question_nature["inference"].append(d)
    elif "comparison" in qa_type:
        grouped_by_question_nature["comparison"].append(d)
    elif "temporal" in qa_type:
        grouped_by_question_nature["temporal"].append(d)


# ---- Evaluation Function ----
def evaluate_group(group_data, model_name):
    total_relevant = 0
    total_retrieved = 0
    total_relevant_retrieved = 0
    ap_scores = []
    gets_a_pass = 0
    bare_minimum_relevant = 0

    for d in group_data:
        relevance = d.get("metadata_relevance", {}).get(model_name, [])
        retrieved_count = len(relevance)
        relevant_count = sum(relevance)

        if relevant_count >= 3:
            gets_a_pass += 1
        if relevant_count == 0:
            bare_minimum_relevant += 1

        if retrieved_count > 0:
            total_retrieved += retrieved_count
            total_relevant_retrieved += relevant_count
        total_relevant += relevant_count

        if relevant_count > 0:
            ap = 0
            relevant_retrieved = 0
            for idx, rel in enumerate(relevance):
                if rel:
                    relevant_retrieved += 1
                    ap += relevant_retrieved / (idx + 1)
            ap /= relevant_count
            ap_scores.append(ap)

    precision = total_relevant_retrieved / total_retrieved if total_retrieved else 0
    mean_ap = sum(ap_scores) / len(ap_scores) if ap_scores else 0

    return precision, mean_ap, bare_minimum_relevant, gets_a_pass


# ---- Evaluate and Print by METADATA TYPE ----
print("\n\n===== Results by Metadata Type =====")
for group_label, entries in grouped_by_metadata_type.items():
    print(f"\nCategory: {group_label}")
    for model in retreival_models:
        p, ap, bm, gp = evaluate_group(entries, model)
        print(f"  {model}: Precision={p:.4f}, MAP={ap:.4f}, BareMin={bm}, GetsPass={gp}")

# ---- Evaluate and Print by QUESTION NATURE ----
print("\n\n===== Results by Question Nature =====")
for group_label, entries in grouped_by_question_nature.items():
    print(f"\nCategory: {group_label}")
    for model in retreival_models:
        p, ap, bm, gp = evaluate_group(entries, model)
        print(f"  {model}: Precision={p:.4f}, MAP={ap:.4f}, BareMin={bm}, GetsPass={gp}")
