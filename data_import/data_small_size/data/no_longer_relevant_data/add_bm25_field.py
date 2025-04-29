import json


with open("/mnt/c/Users/User/thesis/data_import/data_small_size/inital_test_with_big_data.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

for entry in dataset:
    entry["graph_RAG_BM25"] = {
        "answer": "",
        "context": [],
        "cypher_query": ""
    }

with open("/mnt/c/Users/User/thesis/data_import/data_small_size/inital_test_with_big_data.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)