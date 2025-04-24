import json
import pandas as pd


data_path = "/mnt/c/Users/User/thesis/data_import/data_small_size/data/evaluated_dataset.json"

with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

response_counts = {
    'cosine_RAG_response': {'factuality': {'Yes': 0, 'No': 0}, 'relevance': {'Yes': 0, 'No': 0}},
    'graph_RAG_bm25_response': {'factuality': {'Yes': 0, 'No': 0}, 'relevance': {'Yes': 0, 'No': 0}},
    'graph_RAG_cosine_response': {'factuality': {'Yes': 0, 'No': 0}, 'relevance': {'Yes': 0, 'No': 0}}
}

# Loop through the data entries
for entry in data:
    # Extract and count factuality and relevance for cosine_RAG_response
    cosine_rag_response = entry['eval'].get('cosine_RAG_response', '')
    if cosine_rag_response:
        try:
            cosine_data = json.loads(cosine_rag_response)
            if 'factuality' in cosine_data:
                if cosine_data['factuality'] == 'Yes':
                    response_counts['cosine_RAG_response']['factuality']['Yes'] += 1
                elif cosine_data['factuality'] == 'No':
                    response_counts['cosine_RAG_response']['factuality']['No'] += 1
            if 'relevance' in cosine_data:
                if cosine_data['relevance'] == 'Yes':
                    response_counts['cosine_RAG_response']['relevance']['Yes'] += 1
                elif cosine_data['relevance'] == 'No':
                    response_counts['cosine_RAG_response']['relevance']['No'] += 1
        except json.JSONDecodeError:
            pass

    # Extract and count factuality and relevance for graph_RAG_bm25_response
    graph_rag_bm25_response = entry['eval'].get('graph_RAG_bm25_response', '')
    if graph_rag_bm25_response:
        try:
            graph_data = json.loads(graph_rag_bm25_response)
            if 'factuality' in graph_data:
                if graph_data['factuality'] == 'Yes':
                    response_counts['graph_RAG_bm25_response']['factuality']['Yes'] += 1
                elif graph_data['factuality'] == 'No':
                    response_counts['graph_RAG_bm25_response']['factuality']['No'] += 1
            if 'relevance' in graph_data:
                if graph_data['relevance'] == 'Yes':
                    response_counts['graph_RAG_bm25_response']['relevance']['Yes'] += 1
                elif graph_data['relevance'] == 'No':
                    response_counts['graph_RAG_bm25_response']['relevance']['No'] += 1
        except json.JSONDecodeError:
            pass

    # Extract and count factuality and relevance for graph_RAG_cosine_response
    graph_rag_cosine_response = entry['eval'].get('graph_RAG_cosine_response', '')
    if graph_rag_cosine_response:
        try:
            graph_cosine_data = json.loads(graph_rag_cosine_response)
            if 'factuality' in graph_cosine_data:
                if graph_cosine_data['factuality'] == 'Yes':
                    response_counts['graph_RAG_cosine_response']['factuality']['Yes'] += 1
                elif graph_cosine_data['factuality'] == 'No':
                    response_counts['graph_RAG_cosine_response']['factuality']['No'] += 1
            if 'relevance' in graph_cosine_data:
                if graph_cosine_data['relevance'] == 'Yes':
                    response_counts['graph_RAG_cosine_response']['relevance']['Yes'] += 1
                elif graph_cosine_data['relevance'] == 'No':
                    response_counts['graph_RAG_cosine_response']['relevance']['No'] += 1
        except json.JSONDecodeError:
            pass

# Print the results for each response type, separately for "factuality" and "relevance"
print("Response Counts for 'Yes' and 'No':")
for response_type, counts in response_counts.items():
    print(f"\n{response_type}:")
    print(f"  Factuality -> Yes: {counts['factuality']['Yes']}, No: {counts['factuality']['No']}")
    print(f"  Relevance  -> Yes: {counts['relevance']['Yes']}, No: {counts['relevance']['No']}")


# Initialize total counts for each category
total_factuality_yes = 0
total_factuality_no = 0
total_relevance_yes = 0
total_relevance_no = 0
response_types = len(response_counts)

# Sum up the counts for each category (factuality and relevance)
for response_type, counts in response_counts.items():
    total_factuality_yes += counts['factuality']['Yes']
    total_factuality_no += counts['factuality']['No']
    total_relevance_yes += counts['relevance']['Yes']
    total_relevance_no += counts['relevance']['No']

# Calculate the averages for factuality and relevance
avg_factuality_yes = total_factuality_yes / response_types
avg_factuality_no = total_factuality_no / response_types
avg_relevance_yes = total_relevance_yes / response_types
avg_relevance_no = total_relevance_no / response_types

# Calculate the percentage of 'Yes' for each category
percent_factuality_yes = (avg_factuality_yes / (avg_factuality_yes + avg_factuality_no)) * 100
percent_relevance_yes = (avg_relevance_yes / (avg_relevance_yes + avg_relevance_no)) * 100

# Print the results
print(f"Average Factuality - Yes: {avg_factuality_yes}, No: {avg_factuality_no}")
print(f"Percentage Factuality - Yes: {percent_factuality_yes:.2f}%")

print(f"Average Relevance - Yes: {avg_relevance_yes}, No: {avg_relevance_no}")
print(f"Percentage Relevance - Yes: {percent_relevance_yes:.2f}%")

# Initialize an empty dictionary to store aggregated counts for all qa_types
aggregated_data = {}


# Ensure that we process all qa_types and add them if not already present
for entry in data:
    qa_type = entry["qa_type"]
    
     
    # Initialize qa_type in aggregated_data if it does not exist
    if qa_type not in aggregated_data:
        aggregated_data[qa_type] = {
            "cosine_RAG_response": {"factuality": {"Yes": 0, "No": 0}, "relevance": {"Yes": 0, "No": 0}},
            "graph_RAG_bm25_response": {"factuality": {"Yes": 0, "No": 0}, "relevance": {"Yes": 0, "No": 0}},
            "graph_RAG_cosine_response": {"factuality": {"Yes": 0, "No": 0}, "relevance": {"Yes": 0, "No": 0}},
        }

    # Extract the eval data for each response type
    eval_data = entry["eval"]
    
    # Process each response type
    for response_type in ["cosine_RAG_response", "graph_RAG_bm25_response", "graph_RAG_cosine_response"]:
        eval_response = eval_data[response_type].strip()

        # Safely extract factuality and relevance values
        try:
            # Extract factuality if present
            factuality = eval_response.split('"factuality":')[1].split('"')[1] if '"factuality":' in eval_response else None
            
            # Extract relevance if present
            relevance = eval_response.split('"relevance":')[1].split('"')[1] if '"relevance":' in eval_response else None
            
            # Skip this response if either factuality or relevance is missing
            if factuality is None or relevance is None:
                continue
            
            # Update the counts in aggregated_data
            aggregated_data[qa_type][response_type]["factuality"][factuality] += 1
            aggregated_data[qa_type][response_type]["relevance"][relevance] += 1
        except IndexError:
            print(f"Error processing response: {eval_response}")
            continue

#print(aggregated_data)


print("\nAggregated Counts for Each qa_type:")
for qa_type, response_data in aggregated_data.items():
    print(f"qa_type: {qa_type}")
    for response_type, counts in response_data.items():
        factuality_yes = counts["factuality"]["Yes"]
        factuality_no = counts["factuality"]["No"]
        relevance_yes = counts["relevance"]["Yes"]
        relevance_no = counts["relevance"]["No"]
        
        # Output the counts in the desired format
        print(f"  {response_type.replace('_', ' ').title()} - Factuality: {factuality_yes}/{factuality_no}")
        print(f"  {response_type.replace('_', ' ').title()} - Relevance: {relevance_yes}/{relevance_no}")
    print()  # Line break for better readability

# Initialize a list to hold the rows for the detailed DataFrame
detailed_data = []

# Loop through the data entries and aggregate counts for each qa_type
for entry in data:
    qa_type = entry["qa_type"]
    eval_data = entry["eval"]

    # For each response type, extract factuality and relevance and store them in detailed_data
    for response_type in ["cosine_RAG_response", "graph_RAG_bm25_response", "graph_RAG_cosine_response"]:
        eval_response = eval_data.get(response_type, '').strip()

        # Safely extract factuality and relevance values
        try:
            # Extract factuality if present
            factuality = eval_response.split('"factuality":')[1].split('"')[1] if '"factuality":' in eval_response else None
            
            # Extract relevance if present
            relevance = eval_response.split('"relevance":')[1].split('"')[1] if '"relevance":' in eval_response else None
            
            # Skip this response if either factuality or relevance is missing
            if factuality is None or relevance is None:
                continue

            # Append this row to the detailed_data list
            detailed_data.append({
                'qa_type': qa_type,
                'Response Type': response_type,
                'Factuality': factuality,
                'Relevance': relevance
            })
        except IndexError:
            print(f"Error processing response: {eval_response}")
            continue

# Create a DataFrame from the detailed_data list
df_detailed = pd.DataFrame(detailed_data)

# Print the detailed DataFrame
print(df_detailed)

# Optionally, save the detailed DataFrame to an Excel file
df_detailed.to_csv("detailed_output.csv", index=False)