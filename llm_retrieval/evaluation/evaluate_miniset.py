import os
import json
import requests
from datasets import load_dataset

# Load the EdenAI API Key from environment variables
EDENAI_API_KEY = os.getenv("EDENAI_API_KEY")

# Load your dataset (ensure it's in the correct format)
dataset = load_dataset("json", data_files="/mnt/c/Users/User/thesis/data_import/filtered_riksdag.json", split=None)

# Global Message to guide the model behavior
global_message = """
You are a useful model whose job is to evaluate which of two answers is the most relevant.
You will be provided with two answers to one question, as well as the context the question is based on.
You must pick which is the best answer.
The data will be provided to you in this format:
"context": "<p> STYLEREF Kantrubrik \\* MERGEFORMAT Återrapportering från Europeiska rådets möte den 10-11
mars</p><p>Fru talman! Håkan Svenneling tar upp en mycket viktig fråga... </p><p>Återrapporteringen var härmed avslutad.</p>",
"question": "Vad säger Magdalena Andersson om det svenska biståndet till Ryssland?",
"answer1": "Magdalena Andersson klargör att det svenska biståndet till Ryssland inte går till den ryska staten eller den politiska ledningen, utan till civilsamhällesorganisationer vars medlemmar riskerar långa fängelsestraff för att de protesterar mot kriget.",
"answer2": "Magdalena Andersson har uttryckt att Sverige ger ett visst bistånd till Ryssland, men det bör framhållas att detta bistånd inte går till den ryska staten eller den politiska ledningen. Istället går stödet till civilsamhällesorganisationer i Ryssland, vars medlemmar nu riskerar långa fängelsestraff för att de protesterar mot kriget."
Provide your answer like answer like this:
answer1
Do not add any extra information
"""

payload = {
    "providers": "openai/gpt-4o",
    "response_as_dict": True,
    "attributes_as_list": False,
    "show_base_64": True,
    "show_original_response": False,
    "temperature": 0,
    "max_tokens": 4096,
    "tool_choice": "auto",
    "previous_history": dataset, 
    "chatbot_global_action": global_message 
}

headers = {
    "Authorization": f"Bearer {EDENAI_API_KEY}",
    "accept": "application/json",
    "content-type": "application/json"
}

response = requests.post("https://api.edenai.run/v1/models", json=payload, headers=headers)

if response.status_code == 200:
    result = response.json()
    print("Response from EdenAI:", json.dumps(result, indent=4))
else:
    print(f"Error: {response.status_code}")
    print(response.text)  