"""
task4_ann.py
Artificial Neural Network (ANN) using sklearn's MLPClassifier.
We compare two architectures: one hidden layer (64,) and two hidden layers (128, 64).
"""

import json
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from data_prep import (load_and_split, save_figure, plot_confusion_matrix,
                       plot_misclassified, RANDOM_STATE, RESULTS_DIR)

data = load_and_split()
X_train_scaled = data["X_train_scaled"]
X_test_scaled = data["X_test_scaled"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

# ---------- Block 1: train both architectures ----------
architectures = {"(64,)": (64,), "(128, 64)": (128, 64)}
models = {}
rows = []
for name, layers in architectures.items():
    # early_stopping=True keeps 10% of the training data aside as a validation set
    # and stops training when the validation score stops improving (avoids overfitting).
    model = MLPClassifier(hidden_layer_sizes=layers, max_iter=500,
                          early_stopping=True, n_iter_no_change=50,
                          random_state=RANDOM_STATE)
    start = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start

    start = time.time()
    y_pred = model.predict(X_test_scaled)
    predict_time = time.time() - start

    models[name] = model
    rows.append({
        "architecture": name,
        "validation_accuracy": model.best_validation_score_,
        "test_accuracy": accuracy_score(y_test, y_pred),
        "epochs_run": model.n_iter_,
        "train_time_seconds": train_time,
        "predict_time_seconds": predict_time,
    })
    print(f"ANN {name}: test accuracy {rows[-1]['test_accuracy']:.4f}, epochs {model.n_iter_}")

comparison = pd.DataFrame(rows)
comparison.to_csv(os.path.join(RESULTS_DIR, "task4_ann_architecture_comparison.csv"), index=False)

# ---------- Block 2: training loss curves ----------
plt.figure(figsize=(7, 5))
for name, model in models.items():
    plt.plot(model.loss_curve_, label=f"hidden layers {name}")
plt.title("ANN: Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.legend()
plt.grid(alpha=0.3)
save_figure("task4_ann_loss_curve.png")

# ---------- Block 3: choose the best architecture ----------
# We choose using the VALIDATION score, not the test score,
# so the test set stays untouched until the final report.
best_name = comparison.sort_values("validation_accuracy", ascending=False).iloc[0]["architecture"]
best_model = models[best_name]
print("Best architecture (by validation accuracy):", best_name)

y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)   # probability for each of the 10 digits
print(classification_report(y_test, y_pred))

report = classification_report(y_test, y_pred, output_dict=True)
pd.DataFrame(report).T.to_csv(os.path.join(RESULTS_DIR, "task4_ann_classification_report.csv"))

plot_confusion_matrix(y_test, y_pred, f"ANN {best_name} Confusion Matrix (test set)",
                      "task4_ann_confusion_matrix.png")
plot_misclassified(X_test, y_test, y_pred, f"ANN {best_name}: Misclassified Test Images",
                   "task4_ann_misclassified.png")

# Save predictions and probabilities for task 5.
pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(
    os.path.join(RESULTS_DIR, "task4_ann_test_predictions.csv"), index=False)
pd.DataFrame(y_proba, columns=[f"prob_{d}" for d in range(10)]).to_csv(
    os.path.join(RESULTS_DIR, "task4_ann_test_probabilities.csv"), index=False)

# ---------- Block 4: save numbers ----------
best_row = comparison[comparison["architecture"] == best_name].iloc[0]
results = {
    "best_architecture": best_name,
    "test_accuracy": float(best_row["test_accuracy"]),
    "validation_accuracy": float(best_row["validation_accuracy"]),
    "train_time_seconds": float(best_row["train_time_seconds"]),
    "predict_time_seconds": float(best_row["predict_time_seconds"]),
    "epochs_run": int(best_row["epochs_run"]),
    "n_misclassified": int(np.sum(y_test != y_pred)),
    "n_test_images": int(len(y_test)),
}
with open(os.path.join(RESULTS_DIR, "task4_ann_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("Task 4 done.")
