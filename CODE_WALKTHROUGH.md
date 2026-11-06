# CODE WALKTHROUGH (plain-English, block by block)

Every script starts with `from data_prep import ...`, so **start with data_prep.py**.

---

## data_prep.py (shared by everything)

**Block 1: imports and `matplotlib.use("Agg")`**
Agg tells matplotlib to draw into files instead of opening windows. Without it, `run_all.py` would stop at every plot.

**Block 2: settings (`RANDOM_STATE`, folder paths)**
`RANDOM_STATE = 42` is the seed. Anything random (splitting, k-means start points, neural-network starting weights) becomes repeatable. The folder paths are built from the location of the file itself, so the code works from any folder. `os.makedirs(..., exist_ok=True)` makes `figures/` and `results/` if missing.

**Block 3: `load_and_split()`**
- `load_digits()` gives `data` (1797 rows x 64 pixel values) and `target` (the true digit).
- `train_test_split(..., test_size=0.20, stratify=y, random_state=42)` gives 1437 train and 360 test images. **Stratify** means each digit has the same share in both parts (so no digit is accidentally rare in test).
- `idx` is just row numbers. It is carried through the split so we can trace images back if needed.
- `StandardScaler`: makes each pixel have mean 0 and spread 1. **`fit_transform` on train only** learns the mean and spread from training data. `transform` on test only applies those numbers. If we fitted on test too, test information would leak into training (data leakage).
- Returns a dictionary, so each task picks only what it needs by name.

**Block 4: helper functions**
- `save_figure`: saves the current plot to `figures/` at 150 dpi, then closes it (to avoid plots piling up in memory).
- `plot_confusion_matrix`: counts (true digit, predicted digit) pairs and draws them as a heatmap. The diagonal = correct.
- `plot_misclassified`: finds positions where `y_true != y_pred` and draws those images with their true and predicted labels (max 20).

**Viva questions**
1. *Why split before scaling?* So the test set is treated like unseen data. The scaler must learn only from train.
2. *What does stratify do?* Keeps class proportions the same in train and test.
3. *Why random_state=42?* So every run gives the same split and results. The number 42 itself has no special meaning.

---

## task1_pca.py

**Block 1: fit PCA with all 64 components**
PCA finds new directions (components) in the data. The first captures the most variation, the second the next most, and so on. `fit` only on training data.

**Block 2: cumulative variance and the 90% / 95% counts**
`explained_variance_ratio_` = the share of variance each component keeps. `np.cumsum` gives the running total. `np.argmax(cumulative >= 0.90)` finds the first place the total reaches 90% (`+1` because we count from 1, but Python counts from 0). Result: 31 components for 90%, 40 for 95%. The plot shows this curve with dashed lines at 90% and 95%.

**Block 3: 2D scatter**
`transform` projects each image onto the components. We keep only the first two columns (`[:, :2]`) and colour each dot by its true digit.

**Block 4: reconstructions**
For each k in 2, 10, 20, 40: fit PCA with k components, `transform` (compress) the 5 test images, `inverse_transform` (rebuild 64 pixels), then `scaler.inverse_transform` to undo the scaling so the pixel scale is the normal 0-16 again. Fewer components = blurrier picture.

**Block 5: save numbers** to `results/task1_pca_results.json`.

**Viva questions**
1. *What is PCA and why use it?* It turns many correlated features into fewer uncorrelated ones that keep most of the variance. It reduces dimensions, noise and computation time.
2. *What does "95% variance" mean?* The kept components together hold 95% of the total spread of the data. Here, 40 of 64 components.
3. *Why do the 2 component images look so blurry?* The two components hold only 21.7% of the variance, so most detail is thrown away.

---

## task2_kmeans.py

**Block 1: PCA with 95% variance**
`PCA(n_components=0.95)` automatically picks the number of components needed (40). Clustering in fewer dimensions is faster and less noisy.

**Block 2: elbow plot and silhouette for k = 2..15**
For each k we run k-means and record:
- **Inertia** = total squared distance from each point to its cluster centre (lower = tighter, but it always falls as k grows, so we look for an "elbow").
- **Silhouette** = how well each point fits its own cluster compared to the nearest other cluster (from -1 to 1, higher is better).
`n_init=10` means k-means is run 10 times from different starting points and the best run is kept.

**Block 3: k=10, mapping, and metrics**
- `fit_predict` clusters the training images. The labels `y_train` are NOT used here.
- **Mapping:** for each cluster, `np.bincount(...).argmax()` finds the most common true digit inside it. That digit becomes the cluster's name.
- **Purity** = share of images whose cluster name equals their true digit.
- **ARI** (Adjusted Rand Index): agreement between clusters and true labels, corrected for chance. 0 = random, 1 = perfect.
- **NMI** (Normalised Mutual Information): how much knowing the cluster tells you about the digit (0 to 1).
- `predict` on the test set shows how the clusters work on new images.

**Block 4: centroids as images**
Centroids live in PCA space, so we convert back: `pca.inverse_transform`, then `scaler.inverse_transform`. Reshape to 8x8 to view.

**Block 5: heatmap**
`pd.crosstab` counts how many images of each true digit fall in each cluster. A clean result would have one bright cell per row.

**Block 6: save numbers.**

**Viva questions**
1. *How does k-means work?* Pick k centres, assign each point to the nearest centre, move each centre to the mean of its points, repeat until nothing changes.
2. *Why is k-means "unsupervised" yet we report purity?* Labels are never used for clustering. We use them only afterwards to judge the result.
3. *Why is purity (0.669) so much lower than k-NN accuracy (0.967)?* k-means has no labels to learn from, and two digits (8, 9) got no cluster of their own; their images were absorbed into other clusters.

---

## task3_knn.py

**Block 1: GridSearchCV**
We try all 15 values of k x 2 weight options = 30 combinations. Each is scored with 5-fold cross-validation on the training data: the training set is cut into 5 parts, and each part takes a turn as the check set. `weights="distance"` means closer neighbours get a bigger vote.

**Block 2: CV accuracy vs k plot** from `grid.cv_results_` (also saved as CSV).

**Block 3: final model and test**
Train with the best k and weights. `fit` for k-NN just stores the training data, so training is almost instant. Prediction is timed 5 times and averaged because one run is too short to measure reliably. Then the accuracy, classification report (precision / recall / F1 per digit), confusion matrix and misclassified-image grid are made. Predictions are saved for Task 5.

**Block 4: k-NN on PCA features**
PCA (95%) is fitted on train. The same k and weights are used. We compare accuracy and prediction time with the raw features.

**Block 5: save numbers.**

**Viva questions**
1. *How does k-NN classify an image?* It finds the k closest training images (by Euclidean distance) and takes the most common digit among them (a weighted vote if weights="distance").
2. *Why scale the data for k-NN?* Distance-based methods are affected by feature size. Scaling puts all pixels on the same scale.
3. *Why use cross-validation instead of the test set to pick k?* The test set must stay unseen. If we tuned on it, our test accuracy would be too optimistic.

---

## task4_ann.py

**Block 1: train two networks**
`MLPClassifier` is a fully connected neural network. `hidden_layer_sizes=(64,)` = one hidden layer of 64 neurons; `(128, 64)` = two hidden layers. `max_iter=500` = at most 500 passes over the data. `early_stopping=True` = hold out 10% of training data as validation and stop when validation accuracy stops improving (prevents overfitting). Each network's training time, prediction time, and accuracy are recorded.

**Block 2: loss curve**
`loss_curve_` stores the training loss after every epoch. It should fall as the network learns.

**Block 3: choose the best by validation score**
`best_validation_score_` comes from the held-out 10% of training data. We choose with that, not with test accuracy, so the test set stays untouched. (Note: in this run validation picked (64,), but (128, 64) scored higher on test: 0.967 vs 0.944. We report this honestly. The validation set is only about 144 images, so it is noisy.)
`predict_proba` gives a probability for each digit, which Task 5 uses for confidence.

**Block 4: save numbers and predictions.**

**Viva questions**
1. *What is an epoch?* One full pass through all the training data.
2. *What does early stopping do?* It stops training when the score on held-out validation data stops improving, so the network does not memorise the training data.
3. *Why compare (64,) and (128, 64)?* To see whether a bigger/deeper network helps. Here both are close (0.944 vs 0.967), and the small dataset gives only a noisy answer.

---

## task5_comparison.py

**Block 1: load saved results** from tasks 2-4 (no retraining). All models share the same test set, so one `y_true` is enough.

**Block 2: comparison table and bar chart.** For k-means the "accuracy" is purity.

**Block 3: roll-number simulation**
- `rng = np.random.default_rng(42)` makes a repeatable random generator.
- `picked` is a 1000 x 8 table of random test-image row numbers (one roll number per row).
- `pred[picked] == y_true[picked]` compares predictions with truth for all 8000 digits at once. `.all(axis=1)` says "True only if all 8 digits in the row are right".
- **Theory:** if each digit is right with chance p, all 8 are right with chance p^8 (assuming independent digits). Simulated and theory values are close (74.3% vs 76.2% for k-NN).

**Block 4: confidence threshold**
- Digit confidence = the highest probability the ANN gives.
- Roll confidence = the **lowest** digit confidence in the roll (a roll is only as trustworthy as its weakest digit).
- For each threshold from 0.50 to 0.99: rolls above it are auto-accepted, the rest are flagged for a human. We report the % flagged and the accuracy of the accepted ones.

**Block 5: confused pairs**
From the confusion matrix, for each pair (a, b) we add "a read as b" + "b read as a", then sort. Top pair for ANN: 1 and 9 / 1 and 8.

**Viva questions**
1. *Why is roll-number accuracy so much lower than digit accuracy?* One wrong digit ruins the whole roll number: 0.967^8 is about 76%.
2. *What is the trade-off with the confidence threshold?* A higher threshold makes accepted rolls more accurate, but a human has to check more rolls.
3. *Why use the minimum digit confidence per roll?* The whole roll is wrong if any one digit is wrong, so the least-confident digit controls the risk.

---

## run_all.py
Uses `subprocess.run` to start each script one after another with the same Python (`sys.executable`). If any script fails (non-zero return code), it stops, so Task 5 never runs on missing files.

**Viva questions (general)**
1. *Why a separate data_prep.py?* So every model uses the exact same split and scaling, which makes the comparison fair.
2. *Which task is supervised and which is not?* k-NN and ANN are supervised (learn from labels). k-means is unsupervised. PCA is unsupervised too.
3. *What is the main weakness of your results?* The test set is small (360 images), so differences of 1-2% are only a few images and may be chance.
