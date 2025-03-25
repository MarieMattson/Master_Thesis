import json
import random

input_file = "/mnt/c/Users/User/thesis/data_import/filtered_riksdag.json"
output_file = "/mnt/c/Users/User/thesis/data_import/exp2/random_sample.json"

n = 40 

with open(input_file, "r", encoding="utf-8") as file:
    data = json.load(file)

doc_list = ["H90968", "H90982"]
filtered_data = [entry for entry in data if entry.get("dok_id") in doc_list]

seen_anforande_ids = set()
unique_data = []

for entry in filtered_data:
    anforande_id = entry.get("anforande_id")
    if anforande_id and anforande_id not in seen_anforande_ids:
        seen_anforande_ids.add(anforande_id)
        unique_data.append(entry)

random_sample = random.sample(unique_data, min(n, len(unique_data)))

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(random_sample, file, ensure_ascii=False, indent=4)

print(f"Saved {len(random_sample)} random entries to {output_file}")
