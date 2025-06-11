import json
import traceback
from llm_retrieval.FAISS_full_pipeline import FaissRetriever

#/mnt/c/Users/User/thesis/data_import/data_small_size/data/qa_dataset_part1.json
#/mnt/c/Users/User/thesis/data_import/data_small_size/data/qa_dataset_part2.json
#/mnt/c/Users/User/thesis/data_import/data_small_size/data/qa_dataset_part3.json
#/mnt/c/Users/User/thesis/data_import/data_small_size/data/qa_dataset_part4.json
#/mnt/c/Users/User/thesis/data_import/data_small_size/data/null_queries.json


with open("/mnt/c/Users/User/thesis/data_import/data_small_size/data/divided_datasets_for_processing/qa_dataset_part1.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)
print("Script started...")
print(f"Number of entries in dataset: {len(dataset)}")

#temporary jsonl to make it not crash (big data causes problem)
output_path = "/mnt/c/Users/User/thesis/data_import/data_small_size/data/result_faiss_qa_dataset_part1.jsonl"
output_path_json = "/mnt/c/Users/User/thesis/data_import/data_small_size/data/result_faiss_qa_dataset_part1.json"

with open(output_path, "w", encoding="utf-8") as out_file:
    for entry in dataset:
        try:
            question = entry["qa_pair"]["question"]
            
            print(f"\nProcessing question: {question}")
            try:
                ranked_speeches = faiss_rag.retrieve(question)
                entry["cosine_RAG"]["context"] = [speech[0].item() for speech in ranked_speeches]
                entry["cosine_RAG"]["scores"] = [speech[2].item() for speech in ranked_speeches]
                number_of_docs = len(ranked_speeches)
                print(f"Retrieved {number_of_docs} documents.")
            except Exception as e:
                entry["cosine_RAG"]["context"] = f"Retrieval failed: {e}"
                entry["cosine_RAG"]["scores"] = f"Retrieval failed: {e}"
                print(f"Retrieval failed: {e}")
                continue

            try:
                response = faiss_rag.generate_response(ranked_speeches, question)
                print(f"Generated Response:\n{response}")
                entry["cosine_RAG"]["answer"] = response
            except Exception as e:
                entry["cosine_RAG"]["answer"] = f"Response generation failed: {e}"
                print(f"Response generation failed: {e}")
                continue

        except Exception as e:
            print(f"⚠️ Error processing entry: {question}")
            print(traceback.format_exc())

        out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")


with open(output_path, "r", encoding="utf-8") as f_in:
    lines = [json.loads(line) for line in f_in]

with open(output_path_json, "w", encoding="utf-8") as f_out:
    json.dump(lines, f_out, ensure_ascii=False, indent=4)

print("\nProcessing complete!")
