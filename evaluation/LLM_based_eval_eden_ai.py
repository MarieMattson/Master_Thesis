import json
import os
import requests
import time
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()
EDENAI_API_KEY = os.getenv("EDENAI_API_KEY")

# Load and clean input
#with open("/mnt/c/Users/User/thesis/data_import/data_small_size/data/final_248_combined_result.json", "r") as f:
#    dataset = json.load(f)
with open("/mnt/c/Users/User/thesis/data_import/data_large_size/data/full_result_merged.json", "r") as f:
    dataset = json.load(f)

headers = {
    "Authorization": f"Bearer {EDENAI_API_KEY}",
    "accept": "application/json",
    "content-type": "application/json"
}

evaluation_prompt = """
You are an evaluator tasked with judging a questions and responses. 
Your job is to evaluate if the **response** answers the **question**.
---
### Return Yes if:
✅ If the response directly answers the question.
---
### Return No if: 
❌ The response says it cannot answer the question.
❌ The response strays from the question.
---
Return your final evaluation using **only** the following format:
Yes/No
"""

def evaluate_rag_output_with_edenai(question, answer):
    payload = {
        "providers": "google/gemini-1.5-pro-latest",
        "response_as_dict": True,
        "attributes_as_list": False,
        "show_base_64": True,
        "show_original_response": False,
        "temperature": 0,
        "max_tokens": 4096,
        "tool_choice": "auto",
        "previous_history": [
            {"role": "user", "message": f"nQuestion:\n{question}\n\nAnswer:\n{answer}"},
            {"role": "user", "message": evaluation_prompt}
        ]
    }

    try:
        response = requests.post("https://api.edenai.run/v2/text/chat", headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print("EdenAI API error:", response.text)
            return None
    except Exception as e:
        print("Request exception:", e)
        return None

def get_generated_text_safely(evaluation, tag=""):
    try:
        response = evaluation.get("google/gemini-1.5-pro-latest", {})
        if response.get("status") == "success":
            return response.get("generated_text")
        else:
            print(f"[{tag}] Evaluation failed:", response)
    except Exception as e:
        print(f"[{tag}] Exception:", e)

    return None

# Main loop with evaluation
updated_data = []
for idx, entry in enumerate(dataset):
    if "gemini_eval" not in entry:
        entry["gemini_eval"] = {}
    question = entry.get("qa_pair", {}).get("question", "")
    
    for key in ["graph_RAG_cosine", "graph_RAG_bm25", "cosine_RAG"]:
        answer = entry.get(key, {}).get("answer", "") if isinstance(entry.get(key), dict) else entry.get(key, "")
        response_field = f"{key}_response"

        # Only evaluate if field is missing or empty
        if not entry["gemini_eval"].get(response_field):
            print(f"Evaluating [{key}] for entry {idx}")
            result = evaluate_rag_output_with_edenai(question, answer)
            time.sleep(1)  # avoid API rate limit
            verdict = get_generated_text_safely(result, tag=key)
            print(f"→ {response_field}: {verdict}")
            if verdict:
                entry["gemini_eval"][response_field] = verdict

    updated_data.append(entry)

# Save final output
with open("data_import/data_large_size/data/evaluated_full_result_merged.json", "w") as f:
    json.dump(updated_data, f, ensure_ascii=False, indent=2)
