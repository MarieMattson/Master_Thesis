import json
import os
import requests
from datasets import load_dataset
from dotenv import load_dotenv
load_dotenv()
EDENAI_API_KEY = os.getenv("EDENAI_API_KEY")

DATA_FILE_PATH = "/mnt/c/Users/User/thesis/data_import/exp4_improved_prompts/graph_output.json"
OUTPUT_FILE_PATH = "/mnt/c/Users/User/thesis/data_import/exp4_improved_prompts/graph_output.json"
API_URL = "https://api.edenai.run/v2/text/chat"

dataset = load_dataset("json", data_files=DATA_FILE_PATH, split="train")

global_message = """
You are a useful evaluator tasked with determining whether a generated question is valid based on the following guidelines. Review the question and provide a brief explanation for your evaluation. The evaluation should be based on whether the question adheres to the specified rules and guidelines.

Guidelines for Evaluating Validity:

Language:
- The question must be written in Swedish.

Relevance and Focus:
- The question must directly relate to broader political opinions or positions.
- The question should not focus on personal anecdotes or trivial facts unrelated to political discourse.
- The question should reflect political views or positions rather than specific facts or events.

Question Structure:
- The question must inquire about political opinions or positions either of a speaker or a political party.
- Question can ask about the position of the individual or party on a broader political issue (e.g., “Hur argumenterar x för…”).
- Questions can expect a yes or no answer, with a brief explanation provided in the response (e.g., “Anser x att…”).

Avoid Personal or Irrelevant Details:
- Questions should not be based on personal details, unrelated events, or anecdotes. The focus should remain on the political opinions or positions of the speaker or party.

Evaluation Task:
Given the following question, assess whether it follows the guidelines outlined above. Provide a simple "valid" or "invalid".

### Examples:
❌ Disapprove: "Anser Markus Selin att Sverigedemokraternas hållning till ålen är motsägelsefull?"  
   - Too obvious; the question wouldn’t be asked if the answer were “no.”  

❌ Disapprove: "Vad säger Jeanette Gustafsdotter om varför hon inte läser allt som skrivits om henne?"  
   - Focuses on a personal habit rather than a policy or debate.  

✅ Approve: "Vad säger Josefin Malmqvist om yrkeshögskolans roll på den svenska arbetsmarknaden?"  
   - Engages with a broader labor market issue and allows for a substantive response.  

✅ Approve: "Vad är Socialdemokraternas position i frågan om höjning av pensionsavgiften?"  
   - Directly relevant to policy and political debate.  

Use these criteria when evaluating each question. If a question  does not meet the approval standards, disapprove it.

### **Output Format:**  
**Output:::**  
Valid/Invalid  
The question will be provided. If the question meets the criteria, answer "Valid." Otherwise, answer "Invalid."  
Do **NOT** add any explanation, only yes or no.
"""

global_message_temporal = """
You are a useful evaluator tasked with determining whether a generated question is valid based on the following guidelines. Review the question and provide a brief explanation for your evaluation. The evaluation should be based on whether the question adheres to the specified rules and guidelines.

Guidelines for Evaluating Validity:

Language:
- The question must be written in Swedish.

Relevance and Focus:
- The question must directly relate to broader political opinions or positions.
- The question should reflect political views or positions rather than specific facts or events.

Question Structure:
- The question must inquire about specific statements, political opinions or positions either of a speaker or a political party.
- Questions should inquire about the speaker’s or party’s position at different points in time, like a month and year. For example, a question can ask about a speaker's opinion about a certain topic in May 2020.

Evaluation Task:
Given the following question, assess whether it follows the guidelines outlined above. Provide a simple "valid" or "invalid".

Use these criteria when evaluating each question. If a question  does not meet the approval standards, disapprove it.

### **Output Format:**  
**Output:::**  
Valid/Invalid  
The question will be provided. If the question meets the criteria, answer "Valid." Otherwise, answer "Invalid."  
Do **NOT** add any explanation, only yes or no.
"""



# Request headers for EdenAI API
headers = {
    "Authorization": f"Bearer {EDENAI_API_KEY}",
    "accept": "application/json",
    "content-type": "application/json"
}

# Function to evaluate QA pair using EdenAI API
def evaluate_qa_with_edenai(question, question_type):
    if question_type == "generate_qa_temporal":
        payload = {
        "providers": "meta/llama3-1-405b-instruct-v1:0",  # Change provider as needed
        "response_as_dict": True,
        "attributes_as_list": False,
        "show_base_64": True,
        "show_original_response": False,
        "temperature": 0,
        "max_tokens": 4096,
        "tool_choice": "auto",
        "previous_history": [
            {'role': 'user', 'message': f"\nQuestion:\n{question}\n"},
            {'role': 'user', 'message': global_message_temporal}
        ]
    }
    else:
        payload = {
            "providers": "meta/llama3-1-405b-instruct-v1:0",  # Change provider as needed
            "response_as_dict": True,
            "attributes_as_list": False,
            "show_base_64": True,
            "show_original_response": False,
            "temperature": 0,
            "max_tokens": 4096,
            "tool_choice": "auto",
            "previous_history": [
                {'role': 'user', 'message': f"\nQuestion:\n{question}\n"},
                {'role': 'user', 'message': global_message}
            ]
        }

    response = requests.post(API_URL, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error with EdenAI API request: {response.text}")
        return None
    
updated_data = []

for idx, entry in enumerate(dataset):
    question = entry.get("qa_pair", {}).get("question", "N/A")
    question_type = entry.get("qa_type", "N/A")
    evaluation = evaluate_qa_with_edenai(question, question_type)

    if evaluation:
        if "meta/llama3-1-405b-instruct-v1:0" in evaluation and "generated_text" in evaluation["meta/llama3-1-405b-instruct-v1:0"]:
            response = evaluation["meta/llama3-1-405b-instruct-v1:0"]['generated_text']
            entry["LLM_annotator"] = response
            updated_data.append(entry)
        else:
            print(f"Skipping entry {idx} due to content policy violation.")
            entry["LLM_annotator"] = "Content rejected due to policy violation."
            updated_data.append(entry)

    print("*" * 80)

with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as f:
    json.dump(updated_data, f, ensure_ascii=False, indent=4)

print("Dataset successfully updated and saved!")
