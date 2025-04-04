import json

# Path to your JSON file
data_path = "/mnt/c/Users/User/thesis/data_import/exp2/qa_dataset.json"

# Load the data from the JSON file
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Add the new "cosine_RAG" key to each item
for item in data:
    item["graph_RAG"] = {
        "answer": "",
        "context": ""
    }
    item["cosine_RAG"] = {
        "answer": "",
        "context": ""
    }


with open(data_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
print(f"dataset updated and saved to {data_path}")