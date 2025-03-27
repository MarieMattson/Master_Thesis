import json

file_path = "/mnt/c/Users/User/thesis/data_import/exp2/updated_dataset.json"

def transform_json(data):
    """Processes a single JSON object (dictionary)."""
    def normalize_reasonable_question(value):
        if isinstance(value, str) and value.strip().lower() in ["no", "n"]:
            return "No"
        elif isinstance(value, str) and value.strip().lower() in ["yes", "y"]:
            return "Yes"
        return value  # Return unchanged if not in expected values

    if isinstance(data, dict):  # Ensure it's a dictionary before processing
        if "human_annotator" in data and "reasonable_question" in data["human_annotator"]:
            data["human_annotator"]["reasonable_question"] = normalize_reasonable_question(
                data["human_annotator"]["reasonable_question"]
            )
        
        if "LLM_annotator" in data and "reasonable_question" in data["LLM_annotator"]:
            data["LLM_annotator"]["reasonable_question"] = normalize_reasonable_question(
                data["LLM_annotator"]["reasonable_question"]
            )

        # Add RAG_pipeline if not present
        if "RAG_pipeline" not in data:
            data["RAG_pipeline"] = {"answer": "", "context": [], "cypher_query":""}

        data["RAG_pipeline"].setdefault("answer", "")
        data["RAG_pipeline"].setdefault("context", [])
        data["RAG_pipeline"].setdefault("cypher_query", "")
    
    return data

# Load JSON from file
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Determine if JSON is a list or a single dictionary
if isinstance(data, list):
    transformed_data = [transform_json(item) for item in data]  # Process each dictionary in the list
elif isinstance(data, dict):
    transformed_data = transform_json(data)  # Process single dictionary
else:
    print("Error: JSON structure is not recognized.")
    exit(1)

# Save the updated JSON back to the file
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(transformed_data, f, indent=4, ensure_ascii=False)

print("JSON file updated successfully!")
