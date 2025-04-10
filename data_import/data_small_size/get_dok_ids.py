from hashlib import file_digest
import json


with open("/mnt/c/Users/User/thesis/data_import/data_small_size/dataset_small.json", 'r', encoding='utf-8') as f:
        filtered_data = json.load(f)

dok_ids_set = set()

for entry in filtered_data:
        dok_ids_set.add(entry["dok_id"])

print(dok_ids_set)
