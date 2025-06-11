
import json
import random

big_dataset = "/mnt/c/Users/User/thesis/data_import/data_small_size/data/dataset_small.json"
current_results = "/mnt/c/Users/User/thesis/data_import/data_small_size/data/qa_dataset_result.json"
output_file = "/mnt/c/Users/User/thesis/data_import/data_small_size/data/more_dataset_random_entries.json"

with open(current_results, "r", encoding="utf-8") as f:
    current_results = json.load(f)

with open(big_dataset, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract anforande_ids from current_results to avoid duplicates  
anforande_ids = [item.get("anforande_id") for item in current_results if "anforande_id" in item]
anforande_ids = set(anforande_ids)


excluded_titles = {"talmannen", "förste vice talmannen", "andra vice talmannen", "tredje vice talmannen"}
filtered_data = [
    entry for entry in data
    if entry.get("talare", "").lower() not in excluded_titles and entry.get("anforande_id") not in anforande_ids
]
random_entries = random.sample(filtered_data, min(400, len(filtered_data)))

with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(random_entries, file, ensure_ascii=False, indent=4)

print(f"Saved {len(random_entries)} random entries to {output_file}.")
