import json
import traceback
from cosine_rag import CosineRAG

with open("/mnt/c/Users/User/thesis/data_import/exp2/qa_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)
CosineRAG = CosineRAG()


for entry in dataset:
    try:
        question = entry["qa_pair"]["question"]
        print(question)
        
        print(f"\nProcessing question: {question}")
        
        user_query = question

        top_matches = CosineRAG.retrieve(user_query)
        print("Top matches:")
        for anforande_id, match, score in top_matches:
            print(f"Anförande ID: {anforande_id} | Similarity: {score:.4f}")
            print(f"Document: {match[:200]}...\n")

        final_response = CosineRAG.generate_response(top_matches, user_query)

        print("Final Response:", final_response)

        entry["cosine_RAG"]["answer"] = final_response.strip()
        entry["cosine_RAG"]["context"] = entry["cosine_RAG"]["context"] = [anforande_id for anforande_id, _, _ in top_matches]
    
    except Exception as e:
        print(f"⚠️ Error processing entry: {question}")
        print(traceback.format_exc())

with open("/mnt/c/Users/User/thesis/data_import/exp2/qa_dataset_exp2.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print("\nProcessing complete! Results saved to rag_output_exp3.json.")
