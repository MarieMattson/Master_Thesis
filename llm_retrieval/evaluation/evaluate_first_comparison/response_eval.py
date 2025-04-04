import json
from datasets import load_dataset
import os
from dotenv import load_dotenv
import requests

load_dotenv()
EDENAI_API_KEY = os.getenv("EDENAI_API_KEY")
N_EVALUATIONS = 27
dataset = load_dataset("json", data_files="/mnt/c/Users/User/thesis/data_import/exp2/rag_output_qa_dataset_exp2.json", split="train")
url = "https://api.edenai.run/v2/text/chat"

global_message = """
You are an evaluator tasked with determining whether a question-answer pair should be approved or disapproved based on the following criteria. Your goal is to assess if the answer is relevant, informative, and based on the provided context. 

### Approve the question if:
✅ The answer directly addresses the question asked.
✅ The answer is rooted in the provided context and does not introduce external information that was not part of the context.
✅ The question is clear and specific, and the answer provides a well-structured and substantive response to it.
✅ The answer offers relevant details or explanations that contribute to a comprehensive understanding of the topic.
✅ The answer is logically coherent with the context and enhances the discussion of the topic.

### Disapprove the question if:
❌ The answer does not directly respond to the question.
❌ The answer is too vague or lacks substance, offering no meaningful explanation or detail.
❌ The answer contradicts the context or introduces information that is not supported by the context.
❌ The question is too broad or ambiguous, making it impossible to provide a clear, relevant answer.
❌ The question or answer lacks a connection to the context, rendering the answer irrelevant or incomplete.

### **Output Format:**  
 Yes/No

The context, question, and answer will be provided. If the answer directly addresses the question and is grounded in the context, answer "Yes." Otherwise, answer "No."  
Do **NOT** add any explanation, only yes or no.
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
        return evaluation
    else:
        print("Error with EdenAI API request:", response.text)
        return None


print(f"Evaluating {N_EVALUATIONS} QA pairs...")
updated_data = list()

for idx, entry in enumerate(dataset):
    # Initialize the 'eval' key if it doesn't exist
    if 'eval' not in entry:
        entry['eval'] = {}

    # First original qa-pair
    question = entry.get("qa_pair", {}).get("question", "N/A")
    context = entry.get("anforandetext", "N/A")
    original_answer = entry.get("qa_pair", {}).get("answer", "N/A")
    print("original: ", question, "context: ", context, "answer: ", original_answer)
    graph_rag_answer = entry.get("graph_RAG", {}).get("answer", "N/A")
    print("rag_answer: ", graph_rag_answer)
    cosine_rag_answer = entry.get("cosine_RAG", {}).get("answer", "N/A")
    
    # Evaluating original answer
    evaluation_original = evaluate_rag_output_with_edenai(question, original_answer, context)
    evaluation_graph_rag = evaluate_rag_output_with_edenai(question, graph_rag_answer, context)
    evaluation_cosine_rag = evaluate_rag_output_with_edenai(question, graph_rag_answer, context)

    if evaluation_original:
        if "meta/llama3-1-405b-instruct-v1:0" in evaluation_original and "generated_text" in evaluation_original["meta/llama3-1-405b-instruct-v1:0"]:
            response = evaluation_original["meta/llama3-1-405b-instruct-v1:0"]['generated_text']
            entry["eval"]["orig_answer"] = response
            print("evaluation_reasonable_answer_original", response)
        else:
            print(f"Skipping entry {idx} due to content policy violation.")
            entry["eval"]["orig_answer"] = "Content rejected due to policy violation."
    
    print("="*80)
    
    if evaluation_graph_rag:
        if "meta/llama3-1-405b-instruct-v1:0" in evaluation_graph_rag and "generated_text" in evaluation_graph_rag["meta/llama3-1-405b-instruct-v1:0"]:
            response = evaluation_graph_rag["meta/llama3-1-405b-instruct-v1:0"]['generated_text']
            entry["eval"]["graph_RAG_response"] = response
            print("graph_RAG_response", response)
        else:
            print(f"Skipping entry {idx} due to content policy violation.")
            entry["eval"]["graph_RAG_response"] = "Content rejected due to policy violation."
    
    if evaluation_cosine_rag:
        if "meta/llama3-1-405b-instruct-v1:0" in evaluation_cosine_rag and "generated_text" in evaluation_cosine_rag["meta/llama3-1-405b-instruct-v1:0"]:
            response = evaluation_cosine_rag["meta/llama3-1-405b-instruct-v1:0"]['generated_text']
            entry["eval"]["cosine_RAG_response"] = response
            print("cosine_RAG_response", response)
        else:
            print(f"Skipping entry {idx} due to content policy violation.")
            entry["eval"]["cosine_RAG_response"] = "Content rejected due to policy violation."
    
    updated_data.append(entry)


with open("/mnt/c/Users/User/thesis/llm_retrieval/evaluation/evaluate_first_comparison/evaluated_dataset_exp1.json", "w", encoding="utf-8") as f:
    json.dump(updated_data, f, ensure_ascii=False, indent=4)

print("Dataset successfully updated and saved!")
