"""Here I analyse the output of the evaluation from edenai"""

import json


path = "/mnt/c/Users/User/thesis/llm_retrieval/evaluation/evaluate_first_comparison/evaluated_dataset_exp1.json"

with open(path, "r") as f:
    evaluated_data = json.load(f)

orig_score = 0
graph_score = 0
cosine_score = 0

for idx, entry in enumerate(evaluated_data):

    if "Yes" in entry["eval"]["orig_answer"]:# == "Yes":
        orig_score += 1
    if "Yes" in entry["eval"]["graph_RAG_response"]: # == "Yes":
        graph_score += 1
    if "Yes" in entry["eval"]["cosine_RAG_response"]: # == "Yes":
        cosine_score += 1

print(f"Original score: {orig_score/27}")
print(f"Graph score: {graph_score/27}")
print(f"Cosine score: {cosine_score/27}")
