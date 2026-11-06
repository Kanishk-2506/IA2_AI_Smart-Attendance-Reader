# Smart Attendance Reader: handwritten digit recognition

Recognises handwritten digits (like roll numbers on attendance sheets) using PCA, k-means, k-NN and an ANN.
Dataset: `sklearn.datasets.load_digits` (1797 images, 8x8 pixels, 10 classes). Nothing else is downloaded.

## Setup (Windows PowerShell)

```powershell
# 1. Create a virtual environment (a private box for this project's libraries)
py -m venv venv

# 2. Install the libraries
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

On Mac/Linux use `python3 -m venv venv`, then `venv/bin/python -m pip install -r requirements.txt`.

## Run everything

```powershell
.\venv\Scripts\python.exe run_all.py
```

This takes about a minute. It runs the tasks in order and stops if one fails.
You can also run one task alone, e.g. `.\venv\Scripts\python.exe task3_knn.py`.
**Run task 5 only after tasks 2, 3 and 4**, because it reads their saved results.

## Files

| File | What it does |
|---|---|
| `data_prep.py` | Loads data, splits 80/20 (stratified) ONCE, scales (fit on train only). Also has shared plot helpers. |
| `task1_pca.py` | PCA: variance plot, 2D scatter, reconstructions |
| `task2_kmeans.py` | k-means: elbow, silhouette, purity/ARI/NMI, centroids, heatmap |
| `task3_knn.py` | k-NN: grid search, test results, PCA comparison, mistakes |
| `task4_ann.py` | ANN: two architectures, loss curve, test results, mistakes |
| `task5_comparison.py` | Model table, roll-number simulation, confidence thresholds, confused pairs |
| `run_all.py` | Runs all of the above in order |
| `figures/` | All plots (PNG, 150 dpi) |
| `results/` | All numbers (CSV / JSON) |
| `FINDINGS.md` | Bullet-point findings with the real numbers |
| `CODE_WALKTHROUGH.md` | Block-by-block code explanation plus viva questions |

## Settings

`random_state=42` everywhere, so you get the same numbers every run (tiny differences in timings are normal).
