import json
import os
import traceback
import numpy as np
from llm_retrieval.FAISS_full_pipeline import FaissRetriever

# For small data size, you can use the following paths:
# faiss_rag = FaissRetriever(index_path="/mnt/c/Users/User/thesis/data_import/data_small_size/data/index/faiss_index.bin",
#                            anforande_ids_path="/mnt/c/Users/User/thesis/data_import/data_small_size/data/index/anforande_ids.npy",
#                            documents_path="/mnt/c/Users/User/thesis/data_import/data_small_size/data/index/documents.npy"
#                        )

# Paths 
chunk_dir = "/mnt/c/Users/User/thesis/data_import/data_large_size/index/chunked_documents" # only needed for large dataset
input_path = "/mnt/c/Users/User/thesis/data_import/data_small_size/data/null_queries.json"
output_path = "/mnt/c/Users/User/thesis/data_import/data_large_size/data/result_faiss_null_queries.jsonl"
output_path_json = "/mnt/c/Users/User/thesis/data_import/data_large_size/data/result_faiss_null_queries.json"

# skip to initialize FaissRetriever with for small data size (copy paths above)
documents_chunks = []
for i, filename in enumerate(sorted(os.listdir(chunk_dir))):
    # load each chunk and append to documents_chunks list (not concatenated)
    chunk = np.load(os.path.join(chunk_dir, filename), allow_pickle=True, mmap_mode='r')
    documents_chunks.append(chunk)
    

faiss_rag = FaissRetriever(
    index_path="/mnt/c/Users/User/thesis/data_import/data_large_size/index/faiss_index.bin",
    anforande_ids_path="/mnt/c/Users/User/thesis/data_import/data_large_size/index/anforande_ids.npy",
    documents_chunks=documents_chunks
)
print("✅ Retriever initialized")

with open(input_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"📦 Number of entries in dataset: {len(dataset)}")
print("\n🚀 Script started...")
print(f"📦 Number of entries in dataset: {len(dataset)}")

written_lines = 0
with open(output_path, "w", encoding="utf-8") as out_file:
    for i, entry in enumerate(dataset):
        try:
            question = entry["qa_pair"]["question"]
            print(f"\n---\n🔍 Processing question [{i+1}/{len(dataset)}]: {question}")

            if "cosine_RAG" not in entry:
                entry["cosine_RAG"] = {}

            try:
                ranked_speeches = faiss_rag.retrieve(question)
                entry["cosine_RAG"]["context"] = [speech[0].item() for speech in ranked_speeches]
                entry["cosine_RAG"]["scores"] = [speech[2].item() for speech in ranked_speeches]
                print(f"✅ Retrieved {len(ranked_speeches)} documents.")
            except Exception as e:
                entry["cosine_RAG"]["context"] = f"Retrieval failed: {e}"
                entry["cosine_RAG"]["scores"] = f"Retrieval failed: {e}"
                print(f"❌ Retrieval failed: {e}")
                print(traceback.format_exc())
                continue

            try:
                response = faiss_rag.generate_response(ranked_speeches, question)
                entry["cosine_RAG"]["answer"] = response
                print(f"📝 Generated Response:\n{response}")
            except Exception as e:
                entry["cosine_RAG"]["answer"] = f"Response generation failed: {e}"
                print(f"❌ Response generation failed: {e}")
                print(traceback.format_exc())
                continue

        except Exception as e:
            print(f"⚠️ Error processing entry: {e}")
            print(traceback.format_exc())

        out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        written_lines += 1

# Convert JSONL to formatted JSON array
with open(output_path, "r", encoding="utf-8") as f_in:
    lines = [json.loads(line) for line in f_in]

with open(output_path_json, "w", encoding="utf-8") as f_out:
    json.dump(lines, f_out, ensure_ascii=False, indent=4)

print("\n✅ Processing complete!")
print(f"📄 Output JSONL written to: {output_path}")
print(f"🗂 Final JSON written to: {output_path_json}")
print(f"✏️ Total processed entries: {written_lines}")
