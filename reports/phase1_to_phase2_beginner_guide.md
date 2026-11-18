# PathSnap: From Phase 1 to the Final Demo

A beginner-friendly explanation of what we built, why we made the choices we did, how the results should be understood, and how to demonstrate the system.

## 1. The project in one sentence

A phone's GPS readings are noisy and may land beside the road. PathSnap looks at nearby roads for each GPS reading, scores which road is most plausible, and then uses road connectivity to build a route.

A useful picture of the pipeline is:

```text
GPS track + road graph
        ↓
Check and prepare trip data
        ↓
Find 12 nearby road candidates for each GPS point
        ↓
Calculate five features for each candidate
        ↓
Model scores the candidates
        ↓
Viterbi chooses a connected sequence of candidates
        ↓
API returns the prediction, metrics, and map to the UI
```

## 2. What the four dataset files mean

Each trip has four files:

- **`.track`**: the recorded GPS longitude, latitude, and time for the trip.
- **`.nodes`**: the road-network locations where road segments start or end.
- **`.arcs`**: directed road segments connecting pairs of nodes. An arc points from one node to another.
- **`.route`**: the reference sequence of arc IDs for that trip.

The arc IDs are local to a trip. For example, arc 20 in one trip does not mean the same road as arc 20 in a different trip. That's why the classifier learns from geometry and movement features instead of memorizing arc IDs.

The supplied student dataset has **80 training trips** and **10 student-test trips**. The evaluator archive has a separate **10 evaluator trips**.

## 3. What Phase 1 left us with

Phase 1 established a real data-to-prediction pipeline rather than just a written proposal:

1. The first notebooks inspect the files, check coordinates and timestamps, and prepare the road graph.
2. Candidate generation finds up to **12 nearby road arcs** for each GPS point. It uses a spatial KD-tree over road-segment midpoints to find nearby choices efficiently.
3. We calculate **five features** for every GPS-point/candidate-road pair. The exact features used by the code are:
   - **Distance to road (`distance_m`)**: the shortest distance from the GPS point to the road segment. Smaller usually means a closer match.
   - **Heading difference (`heading_difference_rad`)**: how different the direction of movement between GPS points is from the direction of the road segment. Smaller usually means better alignment.
   - **Along-segment fraction (`along_segment_fraction`)**: where the GPS point projects along the segment, from 0 at the start to 1 at the end. This helps distinguish a point near the beginning from one near the end.
   - **Road length (`segment_length_m`)**: the length of the candidate segment.
   - **GPS speed (`gps_speed_mps`)**: distance moved since the previous GPS reading divided by the elapsed time.

   The first three were strongest in the preliminary feature-importance check. We kept all five for the Phase 1 and Phase 2 model comparisons; we did not claim Phase 2 discovered a new feature set.
4. The model-training notebook trains five different classifiers: Logistic Regression, KNN, Decision Tree, Random Forest, and Hist Gradient Boosting.
5. The evaluation notebook compares predictions and shows a route visualization.

Phase 1 used a single trip-wise training/validation split. Hist Gradient Boosting was the initial winner, with a small Top-1 lead over Random Forest in that split. We treated it as a provisional choice because one split is not enough evidence to say a model is always best.

The first Viterbi prototype raised direct road-to-road continuity but lowered point accuracy. That tradeoff was already visible in Phase 1.

### What the five model names mean

- **Logistic Regression:** a simple baseline that combines feature values with learned weights. It is fast and easy to compare against, but its decision boundary is relatively simple.
- **KNN (k-nearest neighbors):** compares a new GPS/candidate example with similar examples seen during training and votes based on those neighbors.
- **Decision Tree:** learns a set of if/then feature splits. It is easy to explain, but one tree can overfit the training examples.
- **Random Forest:** builds many decision trees on varied samples and averages their predictions, usually making a more stable tree model.
- **Hist Gradient Boosting:** builds small trees in sequence, with later trees correcting errors made by earlier ones. It can learn nonlinear relationships among distance, direction, and speed without manually writing those rules.

We do not say Hist Gradient Boosting is automatically the best algorithm for every map-matching problem. It was selected here because it led our declared Phase 2 cross-validation candidate Macro F1 comparison.

## 4. What changed after Phase 1

### A. A more reliable model comparison

Phase 2 replaced the single validation split with **five-fold GroupKFold cross-validation**. In simple terms, the 80 training trips are divided into five rounds. Each round trains on some complete trips and validates on other complete trips. All points from one trip stay together, so near-identical points from a trip cannot appear on both sides of the same round.

The five models were compared using **candidate Macro F1**. The average results were:

| Model | Training-trip CV candidate Macro F1 (mean ± SD) | Student-test Top-1 |
|---|---:|---:|
| Logistic Regression | 0.6358 ± 0.0382 | 57.80% |
| KNN | 0.7202 ± 0.0188 | 66.11% |
| Decision Tree | 0.7981 ± 0.0143 | 73.32% |
| Random Forest | 0.8327 ± 0.0183 | 75.23% |
| **Hist Gradient Boosting** | **0.8485 ± 0.0251** | **75.44%** |

Random Forest had a slightly higher average CV **Top-1** (68.47%) than Hist Gradient Boosting (67.87%). We still chose Hist Gradient Boosting because we selected the model family using the pre-declared CV **candidate Macro F1** objective, where it led. Be precise in the viva: it did not win every metric.

For training labels, each GPS point is associated with the nearest segment in that trip's supplied reference route. The matching road candidate is a positive example and the other candidates are alternatives/negative examples. If the correct reference segment was not among the 12 candidates, that observation cannot provide a positive candidate for the classifier. To keep the training set manageable, each fold samples up to four negative candidates per positive example; KNN uses a smaller balanced sample because storing and comparing all rows is expensive.

### B. Bayesian hyperparameter tuning

After choosing the Hist Gradient Boosting model family, we used **Optuna TPE** to try 15 parameter combinations. TPE is a search method that uses the results of earlier trials to choose later promising trials. It is not a separate prediction model. Every trial used the same trip-grouped training folds, and the target it tried to improve was candidate Macro F1.

The chosen settings were:

- `max_iter = 60`: number of boosting rounds/trees built in sequence.
- `learning_rate = 0.0259`: how much each new boosting round changes the current model.
- `max_leaf_nodes = 31`: maximum leaves in each tree, which limits tree complexity.
- `min_samples_leaf = 47`: minimum training examples required in a leaf.
- `l2_regularization = 0.0487`: a penalty that discourages overly complex model values.

The average CV Macro F1 moved from **0.8485 to 0.8516**. That's a modest improvement, not a dramatic one. On student-test data, candidate Macro F1 rose from 0.8613 to 0.8667, while Top-1 changed slightly down from 75.44% to 75.40%, recall fell from 0.8720 to 0.8348, and log loss worsened from 0.1067 to 0.1352. We kept the tuned model because the declared CV objective improved, and we report the tradeoffs honestly.

The saved final model is `outputs/phase2_model.joblib`.

## 5. What the metrics mean

- **Candidate miss rate**: how often the correct approximate road label was missing from the 12 candidates. The model cannot select a road that candidate generation never offered.
- **Top-1 accuracy**: for each GPS point, did the highest-scoring candidate equal the approximate reference road?
- **Candidate Macro F1**: measures the model's candidate-level correct-versus-alternative decisions while giving both classes a fair role in the score. It is not the same as route-level accuracy.
- **Precision**: of the candidate choices the model marked as correct, how many were correct.
- **Recall**: of the candidates that should be marked correct, how many the model found.
- **Log loss**: checks the quality of the predicted probabilities. Lower is better; a confident wrong probability is penalized heavily.
- **Direct-arc continuity**: how often one chosen road arc connects directly to the next chosen arc. It measures connectedness, not whether every point is assigned to the exact right road.
- **Mean ± SD**: average across the five validation folds plus/minus the standard deviation, a rough indication of how much scores varied between folds.

The route labels themselves are approximated by assigning each GPS point to the nearest segment in the supplied reference route. Crossings and loops can make that approximation ambiguous, so these scores should not be described as perfect ground truth.

## 6. How Viterbi fits into the pipeline

The classifier scores each GPS point separately. A sequence of individually plausible roads can still jump between roads that do not connect.

Viterbi is a dynamic-programming method that considers the whole sequence of candidate choices:

- A **state** is one candidate road arc at one GPS point.
- The **emission score** is the classifier's probability that this candidate fits the GPS point.
- The **transition score** rewards two road arcs that connect in the correct direction and penalizes disconnected choices.
- It stores the best score up to each point and works backward to recover the best-scoring sequence.

Our current decoder uses a simple direct-connectivity rule, not a full road-network shortest-path search. On the student-test set, Top-1 was **75.40% before Viterbi and 56.10% after**, while trip-average direct continuity improved from **25.69% to 77.80%**. It makes routes more connected, but sacrifices point accuracy. Call it a working first decoder, not an accuracy improvement.

## 7. The evaluator API and why labels stay server-side

The teacher's final-evaluation flow requires the evaluator to select any of the 10 reserved trip IDs in the UI. The API then runs the same prediction pipeline and calculates metrics using its internal reference data.

For inference, the API makes a copy of the selected trip with its `.route` labels removed. Candidate generation, the classifier, Viterbi, and map drawing receive that label-free copy. Only after predictions exist does the server-side evaluator read the reference route to calculate evaluation metrics. The API sends the UI the trip ID, predicted arc IDs, aggregate metrics, and map link; it does **not** send the raw reference arc sequence or draw a reference-route layer on the evaluator map.

The API has been run over all 10 evaluator trips after the model was frozen. The aggregate check was:

| Evaluator metric | Classifier output | After Viterbi |
|---|---:|---:|
| Observation-level Top-1 | 69.49% | 54.36% |
| Trip-mean direct-arc continuity | 30.12% | 81.40% |
| Candidate miss rate | 4.79% | — |

These are the final one-time evaluator results, not training or tuning scores. The evaluator result shows the same continuity-versus-accuracy tradeoff as student-test data.

## 8. How to explain the two-laptop architecture

Think of Laptop 1 as the **engine room** and Laptop 2 as the **screen and controls**:

- **Laptop 1:** runs the FastAPI server, loads the saved model and evaluator data, performs prediction, calculates metrics, and prints request logs in its terminal.
- **Laptop 2:** runs the small web server for the HTML UI and displays the trip picker, results, and map in its browser.
- They must be on the same Wi-Fi/LAN. The UI sends a trip ID to Laptop 1's IP address; evaluator data and raw reference routes stay on Laptop 1.

Follow the copyable setup in the project's `README.md`. Before the viva, confirm the API health page reports 10 evaluator trips and that the UI on Laptop 2 connects and lists 10 trip IDs. During evaluation, let the evaluator pick a trip, click **Evaluate trip**, and show the incoming `/api/predict/<trip_id>` request in Laptop 1's terminal as well as the returned result on Laptop 2.

## 9. What files to open during the viva

1. `notebooks/01_Dataset_Exploration.ipynb` — what the files and dataset partitions contain.
2. `notebooks/02_Data_Preprocessing.ipynb` — quality checks and coordinate preparation.
3. `notebooks/03_Candidate_Generation_Features.ipynb` — how candidate roads and the five features are calculated.
4. `notebooks/04_ML_Model_Training.ipynb` — Phase 1 five-model baseline.
5. `notebooks/05_Initial_Evaluation.ipynb` — Phase 1 validation and route decoder.
6. `notebooks/06_Phase2_Model_Validation_Tuning.ipynb` — grouped CV, five-model comparison, Bayesian tuning, student comparison, final model save.
7. `src/pathsnap/server.py` — API endpoints and server-side label separation.
8. `web/index.html` — trip picker, API connection, result cards, and map display.
9. `reports/phase2_working_evidence.md` and `reports/PathSnap_Phase1_Phase2_Technical_Report.pdf` — measured results and limitations across both phases.

## 10. A simple spoken summary

> “We turn each noisy GPS point into a choice among 12 nearby road segments. For each choice, we calculate five geometry and movement features, then train and compare five classifiers. In Phase 2 we kept whole trips together in five-fold cross-validation to avoid mixing points from the same trip between training and validation. Hist Gradient Boosting scored highest on our chosen candidate Macro F1 objective, so we tuned it with 15 Optuna TPE trials. The classifier provides candidate probabilities; a Viterbi decoder uses road connectivity to form a sequence. In the two-laptop demo, Laptop 2 sends the selected trip ID to Laptop 1, where inference and evaluation run. Raw reference routes stay on the API server. Our current Viterbi improves direct continuity but reduces point accuracy, so we present that tradeoff clearly.”

## 11. Important limitations to understand

- **Fifteen** tuning trials were run. This is a small initial search, not an exhaustive search.
- The reference road for each GPS point is approximated from the supplied route geometry.
- Candidate generation misses the approximate reference segment for some points.
- The Viterbi decoder does not yet look for multi-arc shortest paths between observations.
- The tuned model improves the chosen CV objective modestly, but not every student metric.
- The model is a working student-project prototype. Explain the evidence and limitations rather than presenting the scores as perfect.
