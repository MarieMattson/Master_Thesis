import json

input_path = '/mnt/c/Users/User/thesis/data_import/filtered_riksdag.json'
output_path = '/mnt/c/Users/User/thesis/data_import/exp3_cosine_sim/riksdag_wo_bio.json'

# Read JSON file
with open(input_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

def transform_json(entry):
    talare = entry.get("talare", "")
    parti = entry.get("parti", "")
    anforandetext = entry.get("anforandetext", "")

    new_anforandetext = f"{talare} ({parti}):{anforandetext}"

    entry["anforandetext"] = new_anforandetext
    entry.pop("talare", None)
    entry.pop("parti", None)

    return entry

# Transform each entry in the list
transformed_data = [transform_json(entry) for entry in data]

# Save the transformed data to a new JSON file
with open(output_path, 'w', encoding='utf-8') as outfile:
    json.dump(transformed_data, outfile, indent=4, ensure_ascii=False)

print(f"Transformed data saved to {output_path}")
