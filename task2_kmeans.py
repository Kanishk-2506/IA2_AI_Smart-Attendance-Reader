"""
task2_kmeans.py
k-means clustering (unsupervised): the algorithm never sees the true labels
while clustering. We only use labels AFTERWARDS to check how good the clusters are.
"""

import json
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from data_prep import load_and_split, save_figure, RANDOM_STATE, RESULTS_DIR

data = load_and_split()
X_train_scaled = data["X_train_scaled"]
X_test_scaled = data["X_test_scaled"]
y_train = data["y_train"]
y_test = data["y_test"]
scaler = data["scaler"]

# ---------- Block 1: PCA keeping 95% of the variance ----------
# Passing a number between 0 and 1 means "keep enough components for this much variance".
pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"PCA kept {pca.n_components_} components")

# ---------- Block 2: elbow plot and silhouette for k = 2..15 ----------
k_values = list(range(2, 16))
inertias = []      # inertia = sum of squared distances from points to their centroid
silhouettes = []   # silhouette = how well separated the clusters are (higher is better)
for k in k_values:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    km.fit(X_train_pca)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_train_pca, km.labels_))

plt.figure(figsize=(7, 5))
plt.plot(k_values, inertias, marker="o")
plt.title("k-means Elbow Plot (PCA 95% data)")
plt.xlabel("Number of clusters k")
plt.ylabel("Inertia")
plt.grid(alpha=0.3)
save_figure("task2_elbow.png")

plt.figure(figsize=(7, 5))
plt.plot(k_values, silhouettes, marker="o", color="green")
plt.title("k-means Silhouette Score (PCA 95% data)")
plt.xlabel("Number of clusters k")
plt.ylabel("Silhouette score")
plt.grid(alpha=0.3)
save_figure("task2_silhouette.png")

pd.DataFrame({"k": k_values, "inertia": inertias, "silhouette": silhouettes}).to_csv(
    os.path.join(RESULTS_DIR, "task2_elbow_silhouette.csv"), index=False)

# ---------- Block 3: k-means with k = 10 ----------
start = time.time()
kmeans = KMeans(n_clusters=10, n_init=10, random_state=RANDOM_STATE)
train_clusters = kmeans.fit_predict(X_train_pca)
train_time = time.time() - start

# Map each cluster to the digit that appears most often inside it (majority vote).
cluster_to_digit = {}
for c in range(10):
    labels_in_cluster = y_train[train_clusters == c]
    cluster_to_digit[c] = int(np.bincount(labels_in_cluster).argmax())

# Purity = fraction of images whose cluster's majority digit equals their true digit.
mapped_train = np.array([cluster_to_digit[c] for c in train_clusters])
purity_train = float(np.mean(mapped_train == y_train))
ari_train = adjusted_rand_score(y_train, train_clusters)
nmi_train = normalized_mutual_info_score(y_train, train_clusters)

# Also apply the trained clusters to the unseen TEST images.
start = time.time()
test_clusters = kmeans.predict(X_test_pca)
predict_time = time.time() - start
mapped_test = np.array([cluster_to_digit[c] for c in test_clusters])
purity_test = float(np.mean(mapped_test == y_test))

print(f"Train purity={purity_train:.4f}  ARI={ari_train:.4f}  NMI={nmi_train:.4f}")
print(f"Test purity={purity_test:.4f}")
print("Cluster -> majority digit:", cluster_to_digit)

# ---------- Block 4: centroids as 8x8 images ----------
# Centroids live in PCA space, so go back: PCA -> scaled pixels -> original pixels.
centroids_scaled = pca.inverse_transform(kmeans.cluster_centers_)
centroids = scaler.inverse_transform(centroids_scaled)
plt.figure(figsize=(10, 4.5))
for c in range(10):
    plt.subplot(2, 5, c + 1)
    plt.imshow(centroids[c].reshape(8, 8), cmap="gray_r")
    plt.title(f"Cluster {c} (digit {cluster_to_digit[c]})", fontsize=9)
    plt.axis("off")
plt.suptitle("k-means Centroids (k=10) as 8x8 Images")
save_figure("task2_centroids.png")

# ---------- Block 5: cluster vs true digit heatmap ----------
table = pd.crosstab(train_clusters, y_train)   # rows = cluster, columns = true digit
plt.figure(figsize=(8, 6))
sns.heatmap(table, annot=True, fmt="d", cmap="Blues")
plt.title("k-means (k=10): Cluster vs True Digit (training data)")
plt.xlabel("True digit")
plt.ylabel("Cluster number")
save_figure("task2_cluster_vs_digit_heatmap.png")

# ---------- Block 6: save numbers ----------
results = {
    "pca_components_used": int(pca.n_components_),
    "train_purity": purity_train,
    "test_purity": purity_test,
    "adjusted_rand_index": float(ari_train),
    "normalized_mutual_info": float(nmi_train),
    "train_time_seconds": train_time,
    "predict_time_seconds": predict_time,
    "best_silhouette_k": int(k_values[int(np.argmax(silhouettes))]),
    "silhouette_at_k10": float(silhouettes[k_values.index(10)]),
    "cluster_to_digit": {str(c): d for c, d in cluster_to_digit.items()},
    "digits_without_own_cluster": sorted(set(range(10)) - set(cluster_to_digit.values())),
}
with open(os.path.join(RESULTS_DIR, "task2_kmeans_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("Task 2 done.")
