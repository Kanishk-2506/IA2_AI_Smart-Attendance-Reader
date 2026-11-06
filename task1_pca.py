"""
task1_pca.py
PCA: reduce the 64 pixel features to fewer "components" while keeping most information.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from data_prep import (load_and_split, save_figure, RANDOM_STATE, RESULTS_DIR)

data = load_and_split()
X_train_scaled = data["X_train_scaled"]
y_train = data["y_train"]
X_test = data["X_test"]
X_test_scaled = data["X_test_scaled"]
scaler = data["scaler"]

# ---------- Block 1: fit PCA with ALL components ----------
# Fit on training data only. Keeping all 64 lets us see how variance builds up.
pca_full = PCA(n_components=64, random_state=RANDOM_STATE)
pca_full.fit(X_train_scaled)

# Cumulative explained variance: running total of the variance each component keeps.
cumulative = np.cumsum(pca_full.explained_variance_ratio_)

# np.argmax finds the FIRST position where the condition is True.
# +1 because positions start at 0 but we count components from 1.
n_90 = int(np.argmax(cumulative >= 0.90)) + 1
n_95 = int(np.argmax(cumulative >= 0.95)) + 1
print(f"Components for 90% variance: {n_90}")
print(f"Components for 95% variance: {n_95}")

# ---------- Block 2: cumulative variance plot ----------
plt.figure(figsize=(7, 5))
plt.plot(range(1, 65), cumulative, marker="o", markersize=3)
plt.axhline(0.90, color="orange", linestyle="--", label=f"90% variance ({n_90} components)")
plt.axhline(0.95, color="red", linestyle="--", label=f"95% variance ({n_95} components)")
plt.title("PCA: Cumulative Explained Variance")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.legend()
plt.grid(alpha=0.3)
save_figure("task1_cumulative_variance.png")

# ---------- Block 3: 2D scatter of first 2 components ----------
# transform() projects each image onto the principal components.
X_2d = pca_full.transform(X_train_scaled)[:, :2]
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train, cmap="tab10", s=10)
plt.colorbar(scatter, label="Digit")
plt.title("PCA: First Two Components (training data, coloured by digit)")
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
save_figure("task1_pca_2d_scatter.png")

# ---------- Block 4: reconstructions ----------
# For each number of components: compress, then decompress back to 64 pixels.
# The fewer the components, the blurrier the image.
component_counts = [2, 10, 20, 40]
n_images = 5   # show the first 5 test images

plt.figure(figsize=(n_images * 2, 2.2 * (len(component_counts) + 1)))
for col in range(n_images):
    # Top row: the original image
    plt.subplot(len(component_counts) + 1, n_images, col + 1)
    plt.imshow(X_test[col].reshape(8, 8), cmap="gray_r")
    plt.axis("off")
    if col == 0:
        plt.title("Original", fontsize=9, loc="left")

for row, k in enumerate(component_counts):
    pca_k = PCA(n_components=k, random_state=RANDOM_STATE)
    pca_k.fit(X_train_scaled)
    compressed = pca_k.transform(X_test_scaled[:n_images])
    restored_scaled = pca_k.inverse_transform(compressed)
    # Undo the scaling so the pixels go back to the original 0-16 range.
    restored = scaler.inverse_transform(restored_scaled)
    for col in range(n_images):
        plt.subplot(len(component_counts) + 1, n_images, (row + 1) * n_images + col + 1)
        plt.imshow(restored[col].reshape(8, 8), cmap="gray_r")
        plt.axis("off")
        if col == 0:
            plt.title(f"{k} components", fontsize=9, loc="left")
plt.suptitle("PCA Reconstructions: Original vs 2, 10, 20, 40 Components")
save_figure("task1_reconstructions.png")

# ---------- Block 5: save numbers ----------
results = {
    "components_for_90_percent": n_90,
    "components_for_95_percent": n_95,
    "variance_pc1": float(pca_full.explained_variance_ratio_[0]),
    "variance_pc2": float(pca_full.explained_variance_ratio_[1]),
    "cumulative_variance_first_2": float(cumulative[1]),
    "cumulative_variance_at_10": float(cumulative[9]),
    "cumulative_variance_at_20": float(cumulative[19]),
    "cumulative_variance_at_40": float(cumulative[39]),
}
with open(os.path.join(RESULTS_DIR, "task1_pca_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("Task 1 done.")
