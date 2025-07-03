import json
from scipy.stats import ttest_rel

# Paths to your JSON files
large_results_path = "/mnt/c/Users/User/thesis/data_import/data_large_size/data/evaluated_full_result_merged.json"
small_results_path = "/mnt/c/Users/User/thesis/final_248_combined_result_evaluated.json"

def load_binary_scores(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    graph_RAG_cosine = []
    graph_RAG_bm25 = []
    cosine_RAG = []

    for entry in data:
        gemini = entry.get("gemini_eval", {})
        def bin_val(key):
            val = gemini.get(key + "_response", "").strip().lower()
            return 1 if val == "yes" else 0

        graph_RAG_cosine.append(bin_val("graph_RAG_cosine"))
        graph_RAG_bm25.append(bin_val("graph_RAG_bm25"))
        cosine_RAG.append(bin_val("cosine_RAG"))

    return cosine_RAG, graph_RAG_cosine, graph_RAG_bm25

# Load large dataset scores
cosine_RAG_large, graph_RAG_cosine_large, graph_RAG_bm25_large = load_binary_scores(large_results_path)

# Load small dataset scores
cosine_RAG_small, graph_RAG_cosine_small, graph_RAG_bm25_small = load_binary_scores(small_results_path)

# Paired t-tests on large dataset
print("Large Dataset:")
print("Cosine vs Graph+Cosine:", ttest_rel(cosine_RAG_large, graph_RAG_cosine_large))
print("Graph+BM25 vs Graph+Cosine:", ttest_rel(graph_RAG_bm25_large, graph_RAG_cosine_large))
print("Cosine vs Graph+BM25:", ttest_rel(cosine_RAG_large, graph_RAG_bm25_large))

print("\nSmall Dataset:")
# Paired t-tests on small dataset
print("Cosine vs Graph+Cosine:", ttest_rel(cosine_RAG_small, graph_RAG_cosine_small))
print("Graph+BM25 vs Graph+Cosine:", ttest_rel(graph_RAG_bm25_small, graph_RAG_cosine_small))
print("Cosine vs Graph+BM25:", ttest_rel(cosine_RAG_small, graph_RAG_bm25_small))
