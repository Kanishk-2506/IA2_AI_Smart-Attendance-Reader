"""
task5_comparison.py
Compares all models and tests them on a "campus application":
reading 8-digit roll numbers made from handwritten digit images.
This script only READS the results saved by tasks 2, 3 and 4.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from data_prep import save_figure, RANDOM_STATE, RESULTS_DIR

# ---------- Block 1: load saved results ----------
def load_json(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)

knn_res = load_json("task3_knn_results.json")
ann_res = load_json("task4_ann_results.json")
kmeans_res = load_json("task2_kmeans_results.json")

knn_preds = pd.read_csv(os.path.join(RESULTS_DIR, "task3_knn_test_predictions.csv"))
ann_preds = pd.read_csv(os.path.join(RESULTS_DIR, "task4_ann_test_predictions.csv"))
ann_proba = pd.read_csv(os.path.join(RESULTS_DIR, "task4_ann_test_probabilities.csv")).values

y_true = knn_preds["y_true"].values   # same test set for every model
knn_pred = knn_preds["y_pred"].values
ann_pred = ann_preds["y_pred"].values

# ---------- Block 2: comparison table ----------
# For k-means the "accuracy" column is purity (accuracy after mapping clusters to digits).
table = pd.DataFrame([
    {"model": "k-NN", "accuracy": knn_res["test_accuracy"],
     "train_time_s": knn_res["train_time_seconds"], "predict_time_s": knn_res["predict_time_seconds"]},
    {"model": f"ANN {ann_res['best_architecture']}", "accuracy": ann_res["test_accuracy"],
     "train_time_s": ann_res["train_time_seconds"], "predict_time_s": ann_res["predict_time_seconds"]},
    {"model": "k-means (test purity)", "accuracy": kmeans_res["test_purity"],
     "train_time_s": kmeans_res["train_time_seconds"], "predict_time_s": kmeans_res["predict_time_seconds"]},
])
table.to_csv(os.path.join(RESULTS_DIR, "task5_model_comparison.csv"), index=False)
print(table.to_string(index=False))

# Bar chart of the accuracies.
plt.figure(figsize=(7, 5))
plt.bar(table["model"], table["accuracy"], color=["steelblue", "darkorange", "seagreen"])
plt.ylim(0, 1.05)
for i, value in enumerate(table["accuracy"]):
    plt.text(i, value + 0.01, f"{value:.3f}", ha="center")
plt.title("Model Comparison: Test Accuracy (k-means = purity)")
plt.xlabel("Model")
plt.ylabel("Accuracy")
save_figure("task5_model_comparison.png")

# ---------- Block 3: roll-number simulation ----------
# Build 1000 fake roll numbers, each with 8 digits. Each digit is a random TEST image.
rng = np.random.default_rng(RANDOM_STATE)
n_rolls = 1000
n_digits = 8
picked = rng.integers(0, len(y_true), size=(n_rolls, n_digits))   # row numbers of test images

def roll_correct(pred):
    """A roll number is correct only if ALL 8 digits are predicted correctly."""
    digit_ok = (pred[picked] == y_true[picked])   # shape (1000, 8) of True/False
    return digit_ok.all(axis=1)                   # True only when every digit is right

knn_roll_ok = roll_correct(knn_pred)
ann_roll_ok = roll_correct(ann_pred)

# Theory: if each digit is right with probability p, 8 independent digits are all right with p^8.
knn_digit_acc = float(np.mean(knn_pred == y_true))
ann_digit_acc = float(np.mean(ann_pred == y_true))
roll_results = pd.DataFrame([
    {"model": "k-NN", "per_digit_accuracy": knn_digit_acc,
     "roll_accuracy_simulated": float(knn_roll_ok.mean()), "roll_accuracy_theory": knn_digit_acc ** 8},
    {"model": "ANN", "per_digit_accuracy": ann_digit_acc,
     "roll_accuracy_simulated": float(ann_roll_ok.mean()), "roll_accuracy_theory": ann_digit_acc ** 8},
])
roll_results.to_csv(os.path.join(RESULTS_DIR, "task5_roll_number_simulation.csv"), index=False)
print(roll_results.to_string(index=False))

# ---------- Block 4: confidence threshold analysis (ANN) ----------
# Confidence of one digit = the highest probability the ANN gave.
# A roll number is only as trustworthy as its LEAST confident digit.
confidence = ann_proba.max(axis=1)                    # one value per test image
roll_confidence = confidence[picked].min(axis=1)      # one value per roll number

thresholds = np.arange(0.50, 1.00, 0.01)
rows = []
for t in thresholds:
    accepted = roll_confidence >= t           # auto-accepted rolls
    flagged_percent = 100 * (1 - accepted.mean())
    if accepted.sum() > 0:
        accepted_accuracy = 100 * ann_roll_ok[accepted].mean()
    else:
        accepted_accuracy = np.nan            # nothing was accepted
    rows.append({"threshold": round(float(t), 2), "flagged_for_review_percent": flagged_percent,
                 "accuracy_of_accepted_percent": accepted_accuracy})
threshold_table = pd.DataFrame(rows)
threshold_table.to_csv(os.path.join(RESULTS_DIR, "task5_confidence_thresholds.csv"), index=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(threshold_table["threshold"], threshold_table["flagged_for_review_percent"], color="crimson")
ax1.set_title("Roll Numbers Flagged for Human Review")
ax1.set_xlabel("Confidence threshold")
ax1.set_ylabel("Flagged roll numbers (%)")
ax1.grid(alpha=0.3)
ax2.plot(threshold_table["threshold"], threshold_table["accuracy_of_accepted_percent"], color="green")
ax2.set_title("Accuracy of Auto-Accepted Roll Numbers")
ax2.set_xlabel("Confidence threshold")
ax2.set_ylabel("Accuracy of accepted (%)")
ax2.grid(alpha=0.3)
save_figure("task5_confidence_threshold.png")

# ---------- Block 5: most confused digit pairs ----------
def top_confused_pairs(y_t, y_p, n_top=10):
    """Count mistakes between each pair of digits (both directions added together)."""
    cm = confusion_matrix(y_t, y_p)
    pairs = []
    for a in range(10):
        for b in range(a + 1, 10):
            pairs.append({"digit_a": a, "digit_b": b,
                          "a_read_as_b": int(cm[a, b]), "b_read_as_a": int(cm[b, a]),
                          "total_confusions": int(cm[a, b] + cm[b, a])})
    pairs = pd.DataFrame(pairs).sort_values("total_confusions", ascending=False)
    return pairs.head(n_top)

ann_pairs = top_confused_pairs(y_true, ann_pred)
knn_pairs = top_confused_pairs(y_true, knn_pred)
ann_pairs.to_csv(os.path.join(RESULTS_DIR, "task5_confused_pairs_ann.csv"), index=False)
knn_pairs.to_csv(os.path.join(RESULTS_DIR, "task5_confused_pairs_knn.csv"), index=False)
print("Most confused pairs (ANN):")
print(ann_pairs.head(5).to_string(index=False))

top = ann_pairs.head(8)
labels = [f"{a} & {b}" for a, b in zip(top["digit_a"], top["digit_b"])]
plt.figure(figsize=(8, 5))
plt.bar(labels, top["total_confusions"], color="slateblue")
plt.title("Most Confused Digit Pairs (ANN, test set)")
plt.xlabel("Digit pair")
plt.ylabel("Number of confusions (both directions)")
save_figure("task5_confused_pairs.png")
print("Task 5 done.")
