"""
EvalPredictions.py

Compare model predictions against gold labels.

Usage:
    python EvalPredictions.py labels_csv predictions_csv

Positional arguments:
    labels_csv      CSV with columns: Story, id, Sentiments-Aggregated, ...
    predictions_csv CSV with columns: Story, Segment, y_pred, ...

The join is performed on (Story, segment), matching 'id' from labels
with 'Segment' from predictions.
"""

import argparse
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def main(labels_path: str, predictions_path: str) -> None:
    labels = pd.read_csv(labels_path)
    preds = pd.read_csv(predictions_path)

    print(f"Labels rows:      {len(labels)}")
    print(f"Predictions rows: {len(preds)}")

    # Drop multi-label entries from gold labels
    multi = labels["Multi-Aggregated"].str.lower() == "yes"
    if multi.any():
        print(f"Dropping {multi.sum()} row(s) from labels where Multi-Aggregated='yes'.")
    labels = labels[~multi]

    # Rename segment columns to a common key before joining
    labels = labels.rename(columns={"id": "Segment"})

    merged = pd.merge(
        labels[["Story", "Segment", "Sentiments-Aggregated"]],
        preds[["Story", "Segment", "y_pred"]],
        on=["Story", "Segment"],
        how="inner",
    )

    dropped_labels = len(labels) - len(merged)
    dropped_preds = len(preds) - len(merged)

    if dropped_labels > 0:
        print(f"WARNING: {dropped_labels} row(s) from labels had no match in predictions.")
        unmatched = labels[~labels.set_index(["Story", "Segment"]).index.isin(
            merged.set_index(["Story", "Segment"]).index
        )][["Story", "Segment"]]
        print(unmatched.to_string(index=False))

    if dropped_preds > 0:
        print(f"WARNING: {dropped_preds} row(s) from predictions had no match in labels.")
        unmatched = preds[~preds.set_index(["Story", "Segment"]).index.isin(
            merged.set_index(["Story", "Segment"]).index
        )][["Story", "Segment"]]
        print(unmatched.to_string(index=False))

    if dropped_labels == 0 and dropped_preds == 0:
        print("No rows dropped during join.")

    print(f"Merged rows:      {len(merged)}\n")

    # Normalise case
    merged["y_pred"] = merged["y_pred"].str.lower()
    merged["Sentiments-Aggregated"] = merged["Sentiments-Aggregated"].str.lower()

    y_true = merged["Sentiments-Aggregated"]
    y_pred = merged["y_pred"]

    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")

    labels_order = sorted(y_true.unique())
    print("Classification report:")
    print(classification_report(y_true, y_pred, labels=labels_order, zero_division=0))

    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    cm_df = pd.DataFrame(cm, index=labels_order, columns=labels_order)
    print(cm_df.to_string())
    print()

    print("Label distribution — gold vs predicted:")
    dist = pd.DataFrame({
        "gold":      y_true.value_counts(),
        "predicted": y_pred.value_counts(),
    }).fillna(0).astype(int)
    print(dist.to_string())

    # Pearson correlation: map negative→-1, neutral→0, positive→1
    ordinal_map = {"negative": -1, "neutral": 0, "positive": 1}
    y_true_ord = y_true.map(ordinal_map)
    y_pred_ord = y_pred.map(ordinal_map)
    r, p_value = pearsonr(y_true_ord, y_pred_ord)
    print(f"\nPearson correlation: r={r:.4f}, p={p_value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare model predictions against gold labels."
    )
    parser.add_argument("labels_csv", help="CSV with columns: Story, id, Sentiments-Aggregated, ...")
    parser.add_argument("predictions_csv", help="CSV with columns: Story, Segment, y_pred, ...")
    args = parser.parse_args()
    main(args.labels_csv, args.predictions_csv)
