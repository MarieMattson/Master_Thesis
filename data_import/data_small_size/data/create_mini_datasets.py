import json

input_path = "/mnt/c/Users/User/thesis/data_import/data_small_size/data/qa_dataset_175_random_entries.json"
output_path = "/mnt/c/Users/User/thesis/data_import/data_small_size/data/for_final_faiss_retrieval_step_1.json"

# Open the input file and load the dataset
with open(input_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Open the output file for writing the entire list
with open(output_path, "w", encoding="utf-8") as out_file:
    # Write the opening bracket for the JSON array
    out_file.write("[\n")

    # Loop through the entries and dump them with commas between them
    for idx, entry in enumerate(dataset):
        # Dump the entry as JSON
        json.dump(entry, out_file, ensure_ascii=False)
        
        # Add a newline after each entry
        if idx < len(dataset) - 1:  # Don't add a comma after the last entry
            out_file.write(",\n")
        else:
            out_file.write("\n")  # No comma after the last entry

    # Write the closing bracket for the JSON array
    out_file.write("]\n")

print("Processing complete!")
