import pandas as pd


def filter_and_save_by_year(years, input_file, output_file):
    """Filters the dataset based on the given year and saves the result to a JSON file."""
    df = pd.read_csv(input_file, dtype=str)
    filtered_df = df[df['dok_rm'].isin(years)]    
    json_data = filtered_df.to_json(orient="records", force_ascii=False, indent=4)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json_data)

if __name__ == "__main__":
    input_file = "/mnt/c/Users/User/thesis/data_import/full_dataset.csv"
    output_file = "filtered_riksdag.json"
    year_to_filter = ["2021/22"] 

    filter_and_save_by_year(year_to_filter, input_file, output_file)
