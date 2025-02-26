import os
import gzip
import json
import ijson  # To parse JSON incrementally for large files

# Define dataset path
DATA_PATH = "/mnt/c/Users/User/thesis/data_import/data/Data/Riksdagen/dataset.json.gz"

# Ensure the file exists
if not os.path.exists(DATA_PATH):
    print(f"Error: Dataset not found at {DATA_PATH}")
    exit()

# Function to load a small sample of the dataset
def load_sample(file_path, max_items=5):
    """Reads a small sample of the dataset for inspection."""
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        data = json.load(f)  # Fully loads if small
        if isinstance(data, dict):
            items = list(data.items())[:max_items]  # Get first N key-value pairs
            return data, items
        elif isinstance(data, list):
            return data, data[:max_items]  # Get first N elements
        else:
            return data, None

# Function to stream through large JSON data
def stream_json(file_path, max_items=5):
    """Streams a large JSON file incrementally instead of loading it all at once."""
    samples = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        parser = ijson.items(f, '')
        for idx, item in enumerate(parser):
            samples.append(item)
            if idx + 1 >= max_items:  # Stop after retrieving max_items
                break
    return samples

# Try to load a small sample
print(f"Inspecting dataset: {DATA_PATH}")
try:
    dataset, sample_data = load_sample(DATA_PATH, max_items=5)
except MemoryError:
    print("Dataset too large to load fully, switching to streaming method...")
    sample_data = stream_json(DATA_PATH, max_items=5)

# Display dataset metadata
if isinstance(dataset, dict):
    print(f"\nDataset Type: Dictionary with {len(dataset)} keys.")
elif isinstance(dataset, list):
    print(f"\nDataset Type: List with {len(dataset)} elements.")
else:
    print(f"\nDataset Type: {type(dataset)}")

# Display sample data
print("\nSample Data (first 5 entries):")
for item in sample_data:
    print(json.dumps(item, indent=2))  # Pretty-print JSON

# Optionally, save a sample to a file for further inspection
SAMPLE_FILE = "sample_data.json"
with open(SAMPLE_FILE, "w", encoding="utf-8") as f:
    json.dump(sample_data, f, indent=4)
print(f"\nSaved sample data to '{SAMPLE_FILE}' for further inspection.")
