import json
import random

input_file = '/mnt/c/Users/User/thesis/data_import/data_small_size/dataset_small.json'
output_file = '/mnt/c/Users/User/thesis/data_import/data_small_size/dataset_random_entries.json'

with open(input_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

excluded_titles = {"talmannen", "förste vice talmannen", "andra vice talmannen", "tredje vice talmannen"}
filtered_data = [entry for entry in data if entry.get("talare", "").lower() not in excluded_titles]

random_entries = random.sample(filtered_data, min(100, len(filtered_data)))

with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(random_entries, file, ensure_ascii=False, indent=4)

print(f"Saved {len(random_entries)} random entries to {output_file}.")