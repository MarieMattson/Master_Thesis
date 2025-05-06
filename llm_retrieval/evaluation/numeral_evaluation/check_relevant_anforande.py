"""
This is where I check if the anforande is relevant to the question.
This will be done by checking if the anforande is in top 1, 3 or 6.

I can also write code to check if the top anforande is in the same debate as the qa-speech.
"""
from collections import Counter, defaultdict
import json

path = "/mnt/c/Users/User/thesis/data_import/data_small_size/data/final_248_combined_result.json"

with open(path, "r") as f:
    data = json.load(f)

retreival_models = ["graph_RAG_cosine","graph_RAG_bm25", "cosine_RAG"]

def evaluate_relevance(data, retrieval_models):
    for i, d in enumerate(data):
        if "top_anforande_evaluation" not in d:
            d["top_anforande_evaluation"] = {}
        for model in retrieval_models:
            try:
                top_anforande = d[model]["context"][0]
                top_anforande_3 = d[model]["context"][1:4]
                top_anforande_6 = d[model]["context"][1:7]
                print(f"Model: {model}")    

                if d["anforande_id"] in top_anforande:
                    result = "top"
                elif d["anforande_id"] in top_anforande_3:
                    result = "top_3"
                elif d["anforande_id"] in top_anforande_6:
                    result = "top_6"
                else:
                    result = "not_top"
            except Exception as e:
                print(f"Error processing anforande {d['anforande_id']} for model {model}: {e}")
                result = "error"
            
            d["top_anforande_evaluation"][model] = result


evaluate_relevance(data, retreival_models)


output_path = "/mnt/c/Users/User/thesis/data_import/data_small_size/data/evaluated_dataset_with_speech_relevance.json"
with open(output_path, "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

# NOW for counting the results
with open(output_path, "r") as f:
    data = json.load(f)
model_counters = defaultdict(Counter)

# Iterate through the dataset
for d in data:
    evaluations = d.get("top_anforande_evaluation", {})
    for model, result in evaluations.items():
        model_counters[model][result] += 1

# Print out the result nicely
for model, counter in model_counters.items():
    print(f"{model}: ", end="")
    output = ", ".join([f"{k}: {v}" for k, v in counter.items()])
    print(output)