import json

with open("/mnt/c/Users/User/thesis/data_import/filtered_riksdag.json", "r", encoding="utf-8") as file:
    data = json.load(file) 

doc_list = ["H909103", "H90968", "H90982", "H909133", "H90981"]
data = [entry for entry in data if entry.get("dok_id") in doc_list]


with open("filtered_riksdag_exp1.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)  