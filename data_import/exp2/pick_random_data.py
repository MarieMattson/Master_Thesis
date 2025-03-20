import json
import random


input_file = "/mnt/c/Users/User/thesis/data_import/filtered_riksdag.json"
output_file = "/mnt/c/Users/User/thesis/data_import/exp2/random_sample.json"

with open(input_file, "r", encoding="utf-8") as file:
    data = json.load(file)

doc_list = ["H90968", "H90982"]
data = [entry for entry in data if entry.get("dok_id") in doc_list]

n = 30
random_entries = random.sample(data, n)

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(random_entries, file, ensure_ascii=False, indent=4)

print(f"Saved {n} random entries to {output_file}")

