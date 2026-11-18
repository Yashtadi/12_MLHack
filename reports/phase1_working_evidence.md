# Round 1 working evidence

## Data and preprocessing

The supplied archive contains 80 training trips (208,417 GPS observations) and 10 student-test trips (19,160 observations). The evaluator partition was not present. Each trip has four headerless files: `.track` (longitude, latitude, timestamp), `.nodes` (longitude, latitude), `.arcs` (directed node-row pairs), and `.route` (reference arc-row IDs).

The preprocessing notebook shows the schema checks and cleaning decisions. GPS rows with non-finite values, invalid coordinate ranges, or non-increasing timestamps are excluded while original sequence order is preserved. In the inspected partitions there were zero invalid-coordinate rows and zero non-increasing timestamps, so cleaning removed no observations. Gaps over five seconds are retained and reported; speed uses elapsed time. Geographic coordinates are projected to a local metre-scale frame before distance calculations. Road arcs with invalid node references or zero length are excluded from candidate geometry.

## Candidate generation and features

For each GPS point, a KD-tree retrieves the 12 nearest road-segment midpoints. Exact projection onto each candidate segment then provides distance and along-segment position. The notebook displays the projection equations and one worked feature example. Reference labels are assigned to the closest segment in the supplied route. If that segment is absent from the 12 candidates, the observation is recorded as a candidate miss and cannot be recovered by a downstream classifier.

| Feature | Meaning | Reason to include | Permutation importance (mean F1 drop) |
|---|---|---|---:|
| Distance (m) | Point-to-segment perpendicular distance | Main physical proximity cue | 0.1944 |
| Heading difference (rad) | Wrapped difference between recent GPS direction and road direction | Helps distinguish nearby parallel/opposite roads | 0.1166 |
| Along-segment fraction | Clipped projection position between endpoints (0–1) | Indicates whether the point lies along the arc span | 0.1977 |
| Segment length (m) | Candidate arc length | Local geometry context | 0.0054 |
| GPS speed (m/s) | Previous displacement divided by actual elapsed seconds | Movement context, including irregular gaps | 0.0070 |

Permutation importance was measured on 12,000 candidate rows from held-out training trips. It is a first diagnostic, not a causal result or formal ablation. Length and speed contributed little under this setup; the next phase should test feature removal and stronger temporal features.

## Five-model comparison

A fixed random seed reserves 16 whole training trips for validation and fits on the other 64. Keeping whole trips together prevents observations from the same trip appearing in both fit and validation sets. Each method sees the same five features and labels. The nearest-road baseline selects the candidate with minimum distance. For learned models, the positive candidate is the candidate whose arc matches the approximate route label. To reduce class imbalance, training keeps all positive rows and samples up to four negatives per positive (subject to the notebook's cap). Logistic Regression and KNN standardize features; tree methods use the original units.

| Model | Validation Top-1 | Student-test Top-1 | Student-test candidate Macro F1 | Student-test log loss |
|---|---:|---:|---:|---:|
| Nearest road segment | 55.37% | 53.48% | — | — |
| Logistic Regression | 55.22% | 56.50% | 0.6022 | 0.4722 |
| KNN | 60.79% | 65.69% | 0.7251 | 0.5413 |
| Decision Tree | 68.87% | 72.94% | 0.8024 | 0.2230 |
| Random Forest | 71.09% | 75.14% | 0.8414 | 0.1385 |
| Hist Gradient Boosting | **71.45%** | **75.45%** | **0.8609** | **0.1065** |

Top-1 is pointwise: choose the highest-scoring candidate for each GPS point and compare it with the approximate route label. Candidate Macro F1 and log loss are binary candidate-row metrics (correct candidate versus alternative candidate), not a shared global road-ID classification score. Student-test numbers are development comparisons, not evaluator results.

## Why Hist Gradient Boosting was selected

The selection rule was the highest Top-1 accuracy on the held-out training trips. Hist Gradient Boosting achieved 71.45%, just 0.36 percentage points above Random Forest at 71.09%. It also scored slightly higher on the student-test development partition in this run. After selection, the notebook refits the chosen configuration on all 80 training trips and saves `outputs/phase1_model.joblib`. Because the validation margin is small and validation uses one split, this is a provisional choice. Group-aware cross-validation, feature ablation, and tuning are still needed.

## Initial Viterbi result

Across the 10 student-test trips, classifier Top-1 was 75.45% and Viterbi Top-1 was 57.64%. Strict direct-arc continuity increased from 25.41% to 73.86%. The first decoder therefore makes sequences more locally connected while reducing pointwise matches. It is included as a working demonstration and requires multi-arc path costs, elapsed-time-aware transitions, and tuning on permitted development data.

## What the live demo shows

The local interface takes a student-test trip, generates candidates, applies the saved classifier, runs Viterbi, and displays GPS points, road context, reference route, and predicted route on an interactive map. The reference labels are used only for development metrics. Everything runs on one laptop.

## Phase 1 limitations

- No evaluator partition was included.
- The validation is one trip-wise holdout rather than group cross-validation.
- Route labels are approximate nearest-geometry labels.
- Candidate recall limits every model.
- Feature importance is preliminary; no feature-ablation experiment has been completed.
- Viterbi improves direct continuity while lowering Top-1 accuracy.
- Bayesian optimization and final evaluator integration remain later-phase work.
