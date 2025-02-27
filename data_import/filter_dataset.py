import pandas as pd

df = pd.read_csv("/mnt/c/Users/User/thesis/data_import/dataset.csv", dtype=str)

filtered_df = df[df['dok_rm'] == '2012/13']

json_data = filtered_df.to_json(orient="records", force_ascii=False, indent=4)

with open("filtered_riksdag.json", "w", encoding="utf-8") as f:
    f.write(json_data)
