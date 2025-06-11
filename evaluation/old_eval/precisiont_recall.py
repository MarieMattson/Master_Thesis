import json
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_average_precision(gold_standard, retrieved_docs):
    """
    Calculate the Average Precision (AP) for a single query.
    """
    relevant_docs = set(gold_standard)
    retrieved_docs_set = set(retrieved_docs)
    relevant_retrieved_docs = relevant_docs.intersection(retrieved_docs_set)

    if not relevant_retrieved_docs:
        return 0.0  # No relevant documents retrieved
    
    ap = 0.0
    precision_at_k = 0.0
    for k in range(1, len(retrieved_docs) + 1):
        if retrieved_docs[k - 1] in relevant_docs:
            precision_at_k = len(relevant_retrieved_docs.intersection(retrieved_docs[:k])) / k
            ap += precision_at_k
    
    # Normalize AP by the total number of relevant documents
    return ap / len(relevant_docs)

def evaluate_retrieval(dataset, system_key, gold_key="human_annotator", k=6):
    precision_scores = []
    recall_scores = []
    f1_scores = []
    precision_at_k_scores = []
    recall_at_k_scores = []
    average_precision_scores = []

    for entry in dataset:
        gold_standard = entry[gold_key]["context"]
        retrieved_docs = entry[system_key]["context"][:k]
        retrieved_docs_set = set(retrieved_docs)

        # Calculate precision, recall, and f1 score
        all_docs = set(gold_standard).union(retrieved_docs_set)
        y_true = [1 if doc in gold_standard else 0 for doc in all_docs]
        y_pred = [1 if doc in retrieved_docs_set else 0 for doc in all_docs]

        if sum(y_pred) > 0:
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
        else:
            precision, recall, f1 = 0.0, 0.0, 0.0

        # Precision@K
        if len(retrieved_docs) > 0:
            precision_at_k = len(set(gold_standard).intersection(retrieved_docs_set)) / len(retrieved_docs)
        else:
            precision_at_k = 0.0

        # Recall@K
        if len(gold_standard) > 0:
            recall_at_k = len(set(gold_standard).intersection(retrieved_docs_set)) / len(gold_standard)
        else:
            recall_at_k = 0.0

        # Calculate Average Precision for this query
        ap = calculate_average_precision(gold_standard, retrieved_docs)
        average_precision_scores.append(ap)

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        precision_at_k_scores.append(precision_at_k)
        recall_at_k_scores.append(recall_at_k)

    # Compute the MAP (Mean Average Precision)
    mean_ap = sum(average_precision_scores) / len(average_precision_scores)

    return {
        "avg_precision": sum(precision_scores) / len(precision_scores),
        "avg_recall": sum(recall_scores) / len(recall_scores),
        "avg_f1": sum(f1_scores) / len(f1_scores),
        f"avg_precision@{k}": sum(precision_at_k_scores) / len(precision_at_k_scores),
        f"avg_recall@{k}": sum(recall_at_k_scores) / len(recall_at_k_scores),
        "MAP": mean_ap,  # Adding the MAP value
    }

# --- Load dataset and evaluate ---
with open("/mnt/c/Users/User/thesis/data_import/exp2/rag_output_qa_dataset_exp2.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Compare systems
cosine_results = evaluate_retrieval(dataset, system_key="cosine_RAG", k=6)
graph_results = evaluate_retrieval(dataset, system_key="graph_RAG", k=6)

# Print results
def print_results(system_name, results):
    print(f"\n📊 Results for {system_name}:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

print_results("Cosine RAG", cosine_results)
print_results("Graph RAG", graph_results)
