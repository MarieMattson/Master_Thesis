import json
from datasets import load_dataset
import os
from dotenv import load_dotenv
import requests


load_dotenv()
EDENAI_API_KEY = os.getenv("EDENAI_API_KEY")
N_EVALUATIONS = 27
dataset = load_dataset("json", data_files="/mnt/c/Users/User/thesis/data_import/exp2/qa_dataset.json", split="train")
url = "https://api.edenai.run/v2/text/chat"

global_message = """

You are a useful evaluator tasked with determining whether a QA pair is reasonable and should be approved. A question should be approved if it is well-formed, meaningful, and contributes to a relevant discussion. Use the following criteria to decide:

### Approve the question if:
✅ It asks about a broader issue, policy, or debate.  
✅ The answer is not self-evident or redundant based on the question itself.  
✅ It provides meaningful political, social, or economic context.  
✅ The question allows for an informative and substantive answer rather than a simple confirmation or repetition.  
✅ The question is answerable from the provided context.
✅ The question is answered based on the provided context.

### Disapprove the question if:
❌ The answer is too obvious or self-explanatory from the question itself.  
❌ It focuses only on a single statement or phrase without broader relevance.
❌ It lacks political or contextual depth.  
❌ It is vague, unclear, or does not allow for a substantial answer.  
❌ It is about a statement made by the talman (Speaker of the Parliament) — such questions should always be disapproved.  

### Examples:
❌ Disapprove: "Anser Markus Selin att Sverigedemokraternas hållning till ålen är motsägelsefull?"  
   - Too obvious; the question wouldn’t be asked if the answer were “no.”  

❌ Disapprove: "Vad säger Jeanette Gustafsdotter om varför hon inte läser allt som skrivits om henne?"  
   - Focuses on a personal habit rather than a policy or debate.  

❌ Disapprove: "Vad sa talmannen om oppositionens agerande i senaste debatten?"  
   - Questions about statements from the talman should always be disapproved.  

✅ Approve: "Vad säger Josefin Malmqvist om yrkeshögskolans roll på den svenska arbetsmarknaden?"  
   - Engages with a broader labor market issue and allows for a substantive response.  

✅ Approve: "Vad är Socialdemokraternas position i frågan om höjning av pensionsavgiften?"  
   - Directly relevant to policy and political debate.  

Use these criteria when evaluating each QA pair. If a question does not meet the approval standards, disapprove it.


### **Output Format:**  

**Output:::**  

Yes/No

**Output:::**  

The context, question, and answer will be provided. If the question and answer both meet the criteria, answer "Yes." Otherwise, answer "No."  
Do **NOT** add any explanation, only yes or no.
"""


headers = {
    "Authorization": f"Bearer {EDENAI_API_KEY}",
    "accept": "application/json",
    "content-type": "application/json"
}

def evaluate_qa_with_edenai(question, answer, context):
    payload = {
        "providers": "meta/llama3-1-405b-instruct-v1:0", #"openai/gpt-4o",   #"deepseek/DeepSeek-V3",
        "response_as_dict": True,
        "attributes_as_list": False,
        "show_base_64": True,
        "show_original_response": False,
        "temperature": 0,
        "max_tokens": 4096,
        "tool_choice": "auto",
        "previous_history": [
            {'role': 'user', 'message': f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:\n{answer}"}, 
            {'role': 'user', 'message': global_message}
            ]

    }

    response = requests.post(url, json=payload, headers=headers)


    if response.status_code == 200:
        evaluation = response.json()
        return evaluation
    else:
        print("Error with EdenAI API request:", response.text)
        return None


print(f"Evaluating {N_EVALUATIONS} QA pairs...")
updated_data = list()



for idx, entry in enumerate(dataset.select(range(N_EVALUATIONS))):
    question = entry.get("qa_pair", {}).get("question", "N/A")
    answer = entry.get("qa_pair", {}).get("answer", "N/A")
    context = entry.get("anforandetext", "N/A")

    evaluation = evaluate_qa_with_edenai(question, answer, context)
   
    if evaluation:
        if "meta/llama3-1-405b-instruct-v1:0" in evaluation and "generated_text" in evaluation["meta/llama3-1-405b-instruct-v1:0"]:
            response = evaluation["meta/llama3-1-405b-instruct-v1:0"]['generated_text']
            entry["LLM_annotator"]["reasonable_question"] = response
            print(entry)
            updated_data.append(entry)
        else:
            print(f"Skipping entry {idx} due to content policy violation.")
            entry["LLM_annotator"]["reasonable_question"] = "Content rejected due to policy violation."
            updated_data.append(entry)

        print("*" * 80)


with open("/mnt/c/Users/User/thesis/data_import/exp2/updated_dataset.json", "w", encoding="utf-8") as f:
    json.dump(updated_data, f, ensure_ascii=False, indent=4)

print("Dataset successfully updated and saved!")