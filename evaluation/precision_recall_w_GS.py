import json
from sklearn.metrics import precision_score, recall_score, f1_score

k = 6  

with open("/mnt/c/Users/User/thesis/data_import/rag_output_exp1.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

precision_scores = []
recall_scores = []
f1_scores = []
precision_at_k_scores = []
recall_at_k_scores = []

for entry in dataset:
    gold_standard = set(entry["human_annotator"]["context"])
    retrieved_docs = list(entry["RAG_pipeline"]["context"]) 

    retrieved_docs = set(retrieved_docs[:k])
    all_docs = gold_standard.union(retrieved_docs)

    y_true = [1 if doc in gold_standard else 0 for doc in all_docs]
    y_pred = [1 if doc in retrieved_docs else 0 for doc in all_docs]

    if sum(y_pred) > 0:  # Avoid division by zero in precision
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
    else:
        precision, recall, f1 = 0.0, 0.0, 0.0  # If nothing is retrieved

    # Compute Precision@K
    if len(retrieved_docs) > 0:  # Avoid division by zero
        precision_at_k = len(gold_standard.intersection(retrieved_docs)) / len(retrieved_docs)
    else:
        precision_at_k = 0.0

    # Compute Recall@K
    if len(gold_standard) > 0:  # Avoid division by zero
        recall_at_k = len(gold_standard.intersection(retrieved_docs)) / len(gold_standard)
    else:
        recall_at_k = 0.0

    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    precision_at_k_scores.append(precision_at_k)
    recall_at_k_scores.append(recall_at_k)

avg_precision = sum(precision_scores) / len(precision_scores)
avg_recall = sum(recall_scores) / len(recall_scores)
avg_f1 = sum(f1_scores) / len(f1_scores)
avg_precision_at_k = sum(precision_at_k_scores) / len(precision_at_k_scores)
avg_recall_at_k = sum(recall_at_k_scores) / len(recall_at_k_scores)

print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average F1-score: {avg_f1:.4f}")
print(f"Average Precision@{k}: {avg_precision_at_k:.4f}")
print(f"Average Recall@{k}: {avg_recall_at_k:.4f}")
