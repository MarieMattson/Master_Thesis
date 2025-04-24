import json
import time
from datasets import load_dataset
import os
from dotenv import load_dotenv
import requests

load_dotenv()
EDENAI_API_KEY = os.getenv("EDENAI_API_KEY")
dataset = load_dataset("json", data_files="/mnt/c/Users/User/thesis/data_import/data_small_size/data/qa_dataset_result_combined.json", split="train")
url = "https://api.edenai.run/v2/text/chat"

global_message = """
You are an evaluator tasked with judging a question-answer pair based on two distinct criteria: **Factuality** and **Relevance**. 
Your goal is to assess whether the *question* is grounded in the context, and whether the *answer* appropriately responds to the question.

---

### 1. Factuality: Is the answer based on the context?

✅ Yes, if the answer is clearly grounded in information found in the context.  
❌ No, if the answer introduces topics or facts not supported by the context, or if it is unanswerable based on the context.

---

### 2. Relevance: Does the answer respond to the question?

✅ Yes, if the answer directly and specifically addresses the question.  
❌ No, if the answer ignores the question, is vague, off-topic, or introduces irrelevant or contradictory information.

---

### If the answer states that it cannot answer the question, it should be evaluated as follows:
- **Factuality**: ❌ No, because the question is not grounded in the context.
- **Relevance**: ❌ No, because the answer does not respond to the question.

### If unsure, lean toward "No" for either category unless all conditions for "Yes" are clearly met.

---

### Output Format:
Return your final evaluation as a JSON object using **only** the following format:

{ "factuality": "Yes" or "No", "relevance": "Yes" or "No" }

Do not include any other text, explanation, or formatting.
"""

headers = {
    "Authorization": f"Bearer {EDENAI_API_KEY}",
    "accept": "application/json",
    "content-type": "application/json"
}

def evaluate_rag_output_with_edenai(question, answer, context):
    payload = {
        "providers": "meta/llama3-1-405b-instruct-v1:0",  # "openai/gpt-4o",   #"deepseek/DeepSeek-V3",
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
        #print(json.dumps(evaluation, indent=2))  # <-- Add this line to print full LLM response nicely
        return evaluation

    else:
        print("Error with EdenAI API request:", response.text)
        return None

def get_generated_text_safely(evaluation, tag=""):
    try:
        response = evaluation.get("meta/llama3-1-405b-instruct-v1:0", {})
        if response.get("status") != "success":
            print(f"[{tag}] Response status not successful:", response.get("status"))
            print(json.dumps(response, indent=2))
            return None
        return response["generated_text"]
    except Exception as e:
        print(f"[{tag}] Error accessing 'generated_text': {e}")
        print("Full response for debugging:")
        print(json.dumps(evaluation, indent=2))
        return None


updated_data = list()

for idx, entry in enumerate(dataset):
    if 'eval' not in entry:
        entry['eval'] = {}

    # First original qa-pair
    question = entry.get("qa_pair", {}).get("question", "N/A")
    context = entry.get("anforandetext", "N/A")
    answer = entry.get("qa_pair", {}).get("answer", "N/A")
    print("original: ",question) #, "context: ", context,"answer: ", answer)
    
    # Then, the generated RAG_stuff
    graph_rag_cosine_answer = entry.get("graph_RAG_cosine", {}).get("answer", "N/A")
    graph_rag_bm25_answer = entry.get("graph_RAG_bm25", {}).get("answer", "N/A")
    cosine_rag_answer = entry.get("cosine_RAG", {}).get("answer", "N/A")
    print("ragragrag cosine", graph_rag_cosine_answer)    
    print("ragragrag bm25", graph_rag_bm25_answer)
    print("cosinecosinecosine", cosine_rag_answer)
    print("\n","="*80)
    # Evaluating original answer
    evaluation_graph_rag_cosine = evaluate_rag_output_with_edenai(question, graph_rag_cosine_answer, context)
    evaluation_graph_rag_bm25 = evaluate_rag_output_with_edenai(question, graph_rag_bm25_answer, context)
    
    if evaluation_graph_rag_cosine and "meta/llama3-1-405b-instruct-v1:0" in evaluation_graph_rag_cosine:
        verdict = get_generated_text_safely(evaluation_graph_rag_cosine, tag="graph_RAG_cosine")
        if verdict:
            entry["eval"]["graph_RAG_cosine_response"] = verdict
            print("evaluation_graph_rag_cosine:", verdict)
        else:
            print(f"Skipping entry {idx} due to cosine error.")
            entry["eval"]["graph_RAG_cosine_response"] = "Error."
    
    
    if evaluation_graph_rag_bm25 and "meta/llama3-1-405b-instruct-v1:0" in evaluation_graph_rag_bm25:
        verdict = get_generated_text_safely(evaluation_graph_rag_bm25, tag="graph_RAG_bm25")
        if verdict:
            entry["eval"]["graph_RAG_bm25_response"] = verdict
            print("evaluation_reasonable_answer_bm25:", verdict)
        else:
            print(f"Skipping entry {idx} due to bm25 error.")
            entry["eval"]["graph_RAG_bm25_response"] = "Error."
    

    evaluation_cosine_rag = evaluate_rag_output_with_edenai(question, cosine_rag_answer, context)
    if evaluation_cosine_rag and "meta/llama3-1-405b-instruct-v1:0" in evaluation_cosine_rag:
        verdict = get_generated_text_safely(evaluation_cosine_rag, tag="cosine_RAG")
        if verdict:
            entry["eval"]["cosine_RAG_response"] = verdict
            print("evaluation cosine: ", verdict)
        else:
            print(f"Skipping entry {idx} due to cosine RAG error.")
            entry["eval"]["cosine_RAG_response"] = "Error."
    
    updated_data.append(entry)
    time.sleep(0.5)
    print("="*80)

#    print(entry)

with open("/mnt/c/Users/User/thesis/data_import/data_small_size/data/evaluated_dataset.json", "w", encoding="utf-8") as f:
    json.dump(updated_data, f, ensure_ascii=False, indent=4)

print("Dataset successfully updated and saved!")
