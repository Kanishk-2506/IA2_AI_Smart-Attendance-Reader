# FINDINGS (numbers from the actual run, random_state=42)

**Data:** 1797 images (8x8, 10 classes). Stratified 80/20 split gives 1437 train / 360 test. StandardScaler fitted on train only.

## Task 1: PCA
- **Done:** PCA fitted on scaled training data (64 components), cumulative variance plotted, 2D scatter made, images rebuilt from 2/10/20/40 components.
- **Key numbers:**
  - 90% variance needs **31** components; 95% needs **40** (out of 64).
  - PC1 = 12.0% of variance, PC2 = 9.7%, so the first 2 together = 21.7%.
  - Cumulative variance: 10 comps = 59.0%, 20 comps = 79.0%, 40 comps = 95.2%.
- **Figures:** `task1_cumulative_variance.png`, `task1_pca_2d_scatter.png`, `task1_reconstructions.png`.
- **Observation:** Digits are only roughly readable with 10 components, and fairly clear from 20-40.
- **Limitation:** In 2D the digit groups overlap a lot, so 2 components are not enough to separate all digits.
- **Results file:** `results/task1_pca_results.json`

## Task 2: k-means (unsupervised)
- **Done:** k-means on PCA data (95% variance, 40 components), k = 2..15 for elbow and silhouette, then k = 10 with cluster-to-digit majority mapping.
- **Key numbers (k = 10):**
  - Purity: **0.669** (train), **0.661** (test).
  - Adjusted Rand Index: **0.521**. Normalised Mutual Information: **0.667**.
  - Silhouette at k=10: 0.148. The best silhouette is at k=12 (0.170).
  - Digits 8 and 9 got no cluster of their own (two clusters mapped to 7 and two to 4).
- **Figures:** `task2_elbow.png`, `task2_silhouette.png`, `task2_centroids.png`, `task2_cluster_vs_digit_heatmap.png`.
- **Observation:** The elbow plot has no sharp bend, and the silhouette scores are low (0.10-0.17), so the digits do not form clean separate clusters.
- **Limitation:** k-means is fitted on the training set only, and purity depends on the majority-vote mapping. Several clusters share a digit, so purity is much lower than the supervised models.
- **Results files:** `task2_kmeans_results.json`, `task2_elbow_silhouette.csv`

## Task 3: k-NN (supervised)
- **Done:** GridSearchCV (5-fold) over k = 1..15 and weights uniform/distance. The best model was tested on the test set, then repeated on PCA features.
- **Key numbers:**
  - Best settings: **k = 5, weights = distance**; CV accuracy **0.976**.
  - Test accuracy: **0.967** (348 of 360 right, 12 wrong).
  - Weakest classes by recall: digits 8 and 9 (both 0.89).
  - On PCA features (40 comps): test accuracy **0.978**, versus 0.967 on the raw scaled features.
  - Prediction time: 0.0048 s (raw) vs 0.0046 s (PCA), which is about the same at this tiny data size (timings vary run to run).
- **Figures:** `task3_knn_cv_accuracy.png`, `task3_knn_confusion_matrix.png`, `task3_knn_misclassified.png`.
- **Observation:** PCA did not hurt k-NN accuracy (it was 4 images better), so the extra 24 pixel dimensions mostly carried noise.
- **Limitation:** The PCA vs raw gap is only 4 images out of 360, so it may be luck. PCA also did not reduce prediction time here. The PCA model reused k and weights chosen on the raw features.
- **Results files:** `task3_knn_results.json`, `task3_knn_grid_results.csv`, `task3_knn_classification_report.csv`, `task3_knn_test_predictions.csv`

## Task 4: ANN (MLPClassifier)
- **Done:** Two architectures trained (max_iter=500, early_stopping=True, n_iter_no_change=50). The best one was picked by validation score, not test score.
- Default patience (n_iter_no_change=10) stopped training at 34 epochs with 0.944 test accuracy; raising it to 50 let validation accuracy keep improving.
- **Key numbers:**

  | Architecture | Validation acc | Test acc | Epochs | Train time (s) |
  |---|---|---|---|---|
  | (64,) | 0.986 | **0.969** | 162 | 5.20 |
  | (128, 64) | 0.958 | **0.967** | 69 | 4.03 |

  - The validation score picked **(64,)**, which makes 11 mistakes on the 360 test images (was 20 with patience 10).
  - Weakest classes for (64,): digit 8 (recall 0.91) and digit 1 (recall 0.94).
  - With patience 50, (64,) went from 0.965 to 0.986 validation accuracy and from 0.944 to 0.969 test accuracy. (128, 64) stayed about the same (0.967 test).
- **Figures:** `task4_ann_loss_curve.png`, `task4_ann_confusion_matrix.png`, `task4_ann_misclassified.png`.
- **Observation:** The two architectures are now within 1 test image of each other (0.969 vs 0.967), and validation and test agree on which is better. The earlier gap came from training being stopped too early, not from the architecture.
- **Limitation:** The validation set is only about 144 images, so choosing between the two networks by validation score is still noisy. Only one random seed was used. Training is much slower (about 5 s instead of 0.17 s).
- **Results files:** `task4_ann_results.json`, `task4_ann_architecture_comparison.csv`, `task4_ann_classification_report.csv`, `task4_ann_test_predictions.csv`, `task4_ann_test_probabilities.csv`

## Task 5: Comparison and campus application
- **Done:** Compared the three models, simulated 1000 eight-digit roll numbers from test images, analysed confidence thresholds, listed confused digit pairs. The ANN used here is the (64,) model chosen in Task 4.
- **Key numbers: model comparison**

  | Model | Accuracy | Train time (s) | Predict time (s) |
  |---|---|---|---|
  | k-NN | 0.967 | 0.0012 | 0.0048 |
  | ANN (64,) | 0.969 | 5.20 | 0.0029 |
  | k-means (purity) | 0.661 | 0.078 | 0.0020 |

- **Key numbers: roll numbers (1000 simulated, 8 digits each)**

  | Model | Per-digit acc | Simulated roll acc | Theory (acc^8) |
  |---|---|---|---|
  | k-NN | 0.967 | **74.3%** | 76.2% |
  | ANN | 0.969 | **76.2%** | 78.0% |

- **Key numbers: ANN confidence threshold** (a roll is auto-accepted only if its least-confident digit is above the threshold)

  | Threshold | Flagged for review | Accuracy of accepted |
  |---|---|---|
  | 0.50 | 13.5% | 86.7% |
  | 0.60 | 26.4% | 92.1% |
  | 0.70 | 33.3% | 92.1% |
  | 0.80 | 42.4% | 91.7% |
  | 0.90 | 56.5% | 90.3% |
  | 0.95 | 74.1% | 100% |
  | 0.99 | 93.1% | 100% |

- **Key numbers: most confused pairs**
  - ANN: **1 & 8** (3 confusions); every other pair has only 1.
  - k-NN: **1 & 8** (3), then 4 & 7 (2).
- **Figures:** `task5_model_comparison.png`, `task5_confidence_threshold.png`, `task5_confused_pairs.png`.
- **Observation:** One digit error ruins a whole roll number, so about 97% per digit becomes only about 74-76% per roll. The simulation matches theory to within 2 points. The ANN is now slightly ahead of k-NN on both accuracy (0.969 vs 0.967) and roll accuracy (76.2% vs 74.3%), which is a difference of 1 digit and about 2 rolls out of 100, so they are effectively tied. Raising the threshold to 0.95 gives 100% accuracy on accepted rolls, but 74% of rolls then go to a human.
- **Limitation:** Roll numbers are made by sampling the same 360 test images with replacement, so they are not independent real handwriting. Digit counts are tiny (3 confusions for the top pair), and accuracy of accepted rolls does not rise steadily between 0.60 and 0.90 (92.1% to 90.3%), so the "100%" at 0.95+ rests on few rolls. Timings are for a tiny dataset and vary run to run.
- **Results files:** `task5_model_comparison.csv`, `task5_roll_number_simulation.csv`, `task5_confidence_thresholds.csv`, `task5_confused_pairs_ann.csv`, `task5_confused_pairs_knn.csv`
