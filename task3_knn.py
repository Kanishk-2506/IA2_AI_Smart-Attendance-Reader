"""
task3_knn.py
k-Nearest Neighbours (supervised): a new image gets the most common label
among its k closest training images.
"""

import json
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from data_prep import (load_and_split, save_figure, plot_confusion_matrix,
                       plot_misclassified, RESULTS_DIR)

data = load_and_split()
X_train_scaled = data["X_train_scaled"]
X_test_scaled = data["X_test_scaled"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

# ---------- Block 1: grid search with 5-fold cross-validation ----------
# We try every combination of k (1..15) and weights (uniform / distance).
# Cross-validation splits the TRAINING data into 5 parts and rotates which part
# is held out, so we pick the best settings without touching the test set.
param_grid = {"n_neighbors": list(range(1, 16)), "weights": ["uniform", "distance"]}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="accuracy")
grid.fit(X_train_scaled, y_train)
best_k = grid.best_params_["n_neighbors"]
best_weights = grid.best_params_["weights"]
print("Best settings:", grid.best_params_, " CV accuracy:", round(grid.best_score_, 4))

# ---------- Block 2: plot CV accuracy vs k ----------
cv_table = pd.DataFrame(grid.cv_results_)
cv_table[["param_n_neighbors", "param_weights", "mean_test_score", "std_test_score"]].to_csv(
    os.path.join(RESULTS_DIR, "task3_knn_grid_results.csv"), index=False)

plt.figure(figsize=(7, 5))
for w in ["uniform", "distance"]:
    rows = cv_table[cv_table["param_weights"] == w]
    plt.plot(rows["param_n_neighbors"], rows["mean_test_score"], marker="o", label=f"weights = {w}")
plt.title("k-NN: 5-Fold CV Accuracy vs k")
plt.xlabel("Number of neighbours k")
plt.ylabel("Mean CV accuracy")
plt.legend()
plt.grid(alpha=0.3)
save_figure("task3_knn_cv_accuracy.png")

# ---------- Block 3: train the final model and test it ----------
knn = KNeighborsClassifier(n_neighbors=best_k, weights=best_weights)
start = time.time()
knn.fit(X_train_scaled, y_train)
knn_train_time = time.time() - start

# Prediction is timed 5 times and averaged, because one run is too fast to measure well.
times = []
for _ in range(5):
    start = time.time()
    y_pred = knn.predict(X_test_scaled)
    times.append(time.time() - start)
knn_predict_time = float(np.mean(times))

knn_accuracy = accuracy_score(y_test, y_pred)
print(f"k-NN test accuracy (raw scaled features): {knn_accuracy:.4f}")
print(classification_report(y_test, y_pred))

report = classification_report(y_test, y_pred, output_dict=True)
pd.DataFrame(report).T.to_csv(os.path.join(RESULTS_DIR, "task3_knn_classification_report.csv"))

# Save the test predictions so task 5 can reuse them without retraining.
pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(
    os.path.join(RESULTS_DIR, "task3_knn_test_predictions.csv"), index=False)

plot_confusion_matrix(y_test, y_pred, "k-NN Confusion Matrix (test set)",
                      "task3_knn_confusion_matrix.png")
plot_misclassified(X_test, y_test, y_pred, "k-NN: Misclassified Test Images",
                   "task3_knn_misclassified.png")

# ---------- Block 4: k-NN on PCA-reduced features ----------
# Same best k / weights, but only ~40 PCA features instead of 64 pixels.
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

knn_pca = KNeighborsClassifier(n_neighbors=best_k, weights=best_weights)
knn_pca.fit(X_train_pca, y_train)
times = []
for _ in range(5):
    start = time.time()
    y_pred_pca = knn_pca.predict(X_test_pca)
    times.append(time.time() - start)
knn_pca_predict_time = float(np.mean(times))
knn_pca_accuracy = accuracy_score(y_test, y_pred_pca)
print(f"k-NN test accuracy (PCA features, {pca.n_components_} comps): {knn_pca_accuracy:.4f}")

# ---------- Block 5: save numbers ----------
results = {
    "best_k": int(best_k),
    "best_weights": best_weights,
    "best_cv_accuracy": float(grid.best_score_),
    "test_accuracy": float(knn_accuracy),
    "train_time_seconds": knn_train_time,
    "predict_time_seconds": knn_predict_time,
    "n_misclassified": int(np.sum(y_test != y_pred)),
    "n_test_images": int(len(y_test)),
    "pca_components": int(pca.n_components_),
    "pca_test_accuracy": float(knn_pca_accuracy),
    "pca_predict_time_seconds": knn_pca_predict_time,
}
with open(os.path.join(RESULTS_DIR, "task3_knn_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("Task 3 done.")
