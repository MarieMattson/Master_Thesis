import json

input_path = "/mnt/c/Users/User/thesis/data_import/exp4_improved_prompts/graph_output.json"
output_path = "/mnt/c/Users/User/thesis/data_import/exp4_improved_prompts/graph_output.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for entry in data:
    entry["LLM_annotator"] = ""

    entry["graph_RAG"] = {
        "answer": "",
        "context": [],
        "cypher_query": ""
    }

    entry["cosine_RAG"] = {
        "answer": "",
        "context": []
    }

    entry["eval"] = {
        "orig_answer": "",
        "graph_RAG_response": "",
        "cosine_RAG_response": ""
    }

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
