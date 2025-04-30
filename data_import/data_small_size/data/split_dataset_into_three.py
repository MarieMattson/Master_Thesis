import json
import os

# Input file path
input_file = '/mnt/c/Users/User/thesis/data_import/data_small_size/data/qa_dataset_175_random_entries.json'

# Output file paths
output_file_1 = '/mnt/c/Users/User/thesis/data_import/data_small_size/data/qa_dataset_part1.json'
output_file_2 = '/mnt/c/Users/User/thesis/data_import/data_small_size/data/qa_dataset_part2.json'
output_file_3 = '/mnt/c/Users/User/thesis/data_import/data_small_size/data/qa_dataset_part3.json'

# Load the dataset
with open(input_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Calculate the size of each split
total_entries = len(data)
split_size = total_entries // 3

# Split the dataset
part1 = data[:split_size]
part2 = data[split_size:2 * split_size]
part3 = data[2 * split_size:]

with open(output_file_1, 'w', encoding='utf-8') as file:
    json.dump(part1, file, indent=4, ensure_ascii=False)

with open(output_file_2, 'w', encoding='utf-8') as file:
    json.dump(part2, file, indent=4, ensure_ascii=False)

with open(output_file_3, 'w', encoding='utf-8') as file:
    json.dump(part3, file, indent=4, ensure_ascii=False)


print(f"Dataset split into three parts and saved to:\n{output_file_1}\n{output_file_2}\n{output_file_3}")