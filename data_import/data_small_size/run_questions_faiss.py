import json
import traceback
from llm_retrieval.retrieve_from_faiss import FaissRetriever

faiss_rag = FaissRetriever(index_path="/mnt/c/Users/User/thesis/data_import/data_small_size/data/index/faiss_index.bin",
                            anforande_ids_path="/mnt/c/Users/User/thesis/data_import/data_small_size/data/index/anforande_ids.npy",
                            documents_path="/mnt/c/Users/User/thesis/data_import/data_small_size/data/index/documents.npy"
                        )
with open("/mnt/c/Users/User/thesis/data_import/data_small_size/data/qa_dataset_175_random_entries.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)
print("Script started...")
print(f"Number of entries in dataset: {len(dataset)}")
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

with open("/mnt/c/Users/User/thesis/data_import/data_small_size/data/final_175_faiss_result.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print("\nProcessing complete!")
