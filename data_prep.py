"""
data_prep.py
Shared file used by ALL tasks.
It loads the digits dataset, splits it ONCE into train/test, and scales it.
It also holds two small plotting helpers that tasks 3 and 4 both use.
"""

import os
import numpy as np
import matplotlib

# "Agg" means: draw plots straight into files, do not open pop-up windows.
# This lets run_all.py run without stopping at every plot.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------- Settings used everywhere ----------
RANDOM_STATE = 42   # same seed everywhere so results can be repeated

# Folders are built from the location of this file, so the scripts work
# no matter which folder you run them from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create the folders if they do not exist yet.
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_and_split():
    """
    Load the digits data, split it 80/20 (stratified), and scale it.
    Returns a dictionary so every task can pick what it needs by name.
    """
    digits = load_digits()
    X = digits.data     # shape (1797, 64): each row is one 8x8 image flattened
    y = digits.target   # shape (1797,): the true digit 0-9

    # Stratified split: each digit keeps the same share in train and test.
    # We also keep the row numbers (idx) so we can find the original images later.
    idx = np.arange(len(y))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, idx, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # Scaling: fit ONLY on the training data (so no information leaks from test),
    # then use the same scaler to transform both train and test.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train": X_train, "X_test": X_test,                  # raw pixels (0-16)
        "X_train_scaled": X_train_scaled,                      # scaled pixels
        "X_test_scaled": X_test_scaled,
        "y_train": y_train, "y_test": y_test,
        "scaler": scaler,
    }


def save_figure(filename):
    """Save the current figure into figures/ at 150 dpi, then close it."""
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, title, filename):
    """Draw a confusion matrix heatmap (rows = true digit, columns = predicted digit)."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted digit")
    plt.ylabel("True digit")
    save_figure(filename)


def plot_misclassified(X_test_raw, y_true, y_pred, title, filename, max_images=20):
    """Show a grid of test images the model got wrong, with true vs predicted label."""
    wrong = np.where(y_true != y_pred)[0]   # positions of the mistakes
    wrong = wrong[:max_images]              # show at most max_images of them

    n_cols = 5
    n_rows = max(1, int(np.ceil(len(wrong) / n_cols)))
    plt.figure(figsize=(n_cols * 2, n_rows * 2.2))
    for i, position in enumerate(wrong):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(X_test_raw[position].reshape(8, 8), cmap="gray_r")
        plt.title(f"True {y_true[position]}, Pred {y_pred[position]}", fontsize=9)
        plt.axis("off")
    plt.suptitle(title)
    save_figure(filename)
