# Currently does not work!
from datasets import load_dataset
from torch.utils.data import DataLoader

dataset = load_dataset("KBLab/rixvox", cache_dir="data_rixvox", streaming=True)
dataset_valid = dataset["validation"].take(5)

for example in dataset_valid:
    print(example)