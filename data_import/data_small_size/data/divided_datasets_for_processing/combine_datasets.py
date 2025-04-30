import json

# Load your two datasets
with open('/mnt/c/Users/User/thesis/data_import/data_small_size/data/divided_datasets_for_processing/full_result_faiss_qa_dataset.json', 'r', encoding='utf-8') as f:
    data1 = json.load(f)

with open('/mnt/c/Users/User/thesis/data_import/data_small_size/data/divided_datasets_for_processing/full_result_graph_qa_dataset.json', 'r', encoding='utf-8') as f:
    data2 = json.load(f)

# Index data2 by anforande_id for quick lookup
data2_index = {item['anforande_id']: item for item in data2}

def merge_prioritize_graph_fields(base, update):
    result = base.copy()

    # Prioritize these fields from data2
    for key in ["graph_RAG_cosine", "graph_RAG_bm25"]:
        if key in update:
            result[key] = update[key]

    # Merge remaining fields, keeping base unless update has a better value
    for key, value in update.items():
        if key not in ["graph_RAG_cosine", "graph_RAG_bm25"]:
            if (
                key in result and isinstance(result[key], dict) and isinstance(value, dict)
            ):
                result[key] = merge_prioritize_graph_fields(result[key], value)
            elif value not in [None, "", [], {}]:
                result[key] = value
    return result

# Merge the datasets
merged_data = []
for item in data1:
    anforande_id = item['anforande_id']
    if anforande_id in data2_index:
        merged_item = merge_prioritize_graph_fields(item, data2_index[anforande_id])
        merged_data.append(merged_item)
    else:
        merged_data.append(item)

# Save to new file
with open('/mnt/c/Users/User/thesis/data_import/data_small_size/data/divided_datasets_for_processing/final_full_merged_data.json', 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, indent=4, ensure_ascii=False)
