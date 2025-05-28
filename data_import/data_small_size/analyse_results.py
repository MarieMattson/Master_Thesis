from collections import defaultdict
import json

data_path = "data_import/data_large_size/data/evaluated_full_result_merged.json"

with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Main model response counters
cosine_RAG_response_dict = defaultdict(int)
graph_RAG_bm25_response_dict = defaultdict(int)
graph_RAG_cosine_response_dict = defaultdict(int)

# QA-type-specific response counters
cosine_RAG_by_qa_type = defaultdict(lambda: defaultdict(int))
graph_RAG_bm25_by_qa_type = defaultdict(lambda: defaultdict(int))
graph_RAG_cosine_by_qa_type = defaultdict(lambda: defaultdict(int))

for entry in data:
    qa_type = entry.get("qa_type", "unknown")

    cosine_RAG_response = entry['gemini_eval'].get('cosine_RAG_response', '').strip()
    cosine_RAG_response_dict[cosine_RAG_response] += 1
    cosine_RAG_by_qa_type[qa_type][cosine_RAG_response] += 1

    graph_rag_bm25_response = entry['gemini_eval'].get('graph_RAG_bm25_response', '').strip()
    graph_RAG_bm25_response_dict[graph_rag_bm25_response] += 1
    graph_RAG_bm25_by_qa_type[qa_type][graph_rag_bm25_response] += 1

    graph_rag_cosine_response = entry['gemini_eval'].get('graph_RAG_cosine_response', '').strip()
    graph_RAG_cosine_response_dict[graph_rag_cosine_response] += 1
    graph_RAG_cosine_by_qa_type[qa_type][graph_rag_cosine_response] += 1

# Print global model response counts
print("=== Overall Response Counts ===")
print("Graph RAG (Cosine):", dict(graph_RAG_cosine_response_dict))
print("Graph RAG (BM25):", dict(graph_RAG_bm25_response_dict))
print("Cosine RAG:", dict(cosine_RAG_response_dict))

# Print response counts broken down by QA type
print("\n=== Response Counts by QA Type ===")
print("Cosine RAG by QA Type:")
for qa, responses in cosine_RAG_by_qa_type.items():
    print(f"{qa}: {dict(responses)}")

print("\nGraph RAG (BM25) by QA Type:")
for qa, responses in graph_RAG_bm25_by_qa_type.items():
    print(f"{qa}: {dict(responses)}")

print("\nGraph RAG (Cosine) by QA Type:")
for qa, responses in graph_RAG_cosine_by_qa_type.items():
    print(f"{qa}: {dict(responses)}")
