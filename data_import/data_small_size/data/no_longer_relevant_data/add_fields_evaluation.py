import json

# Load your dataset
with open('/mnt/c/Users/User/thesis/data_import/data_small_size/data/evaluated_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Make sure data is a list of items or a dict
if isinstance(data, dict):
    data = [data]

# Update each item
for item in data:
    item['openai_eval'] = {
        "cosine_RAG_response": "",
        "graph_RAG_bm25_response": "",
        "graph_RAG_cosine_response": ""
    }
    item['gemini_eval'] = {
        "cosine_RAG_response": "",
        "graph_RAG_bm25_response": "",
        "graph_RAG_cosine_response": ""
    }

# Save the updated dataset
with open('/mnt/c/Users/User/thesis/data_import/data_small_size/data/evaluated_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ Dataset updated successfully!")
