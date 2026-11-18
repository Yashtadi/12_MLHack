# Phase 1 working evidence

## Preprocessing performed

1. Read the four headerless per-trip files using fixed schemas: GPS (longitude, latitude, timestamp), nodes (longitude, latitude), arcs (from node, to node), and route arc-row IDs.
2. Drop GPS rows with non-finite values or coordinates outside valid longitude/latitude bounds. Preserve observation order and drop duplicate/backward timestamps rather than sorting a trajectory.
3. Keep GPS gaps in the sequence; compute speed using the actual elapsed timestamp so a gap is not mistaken for a one-second step.
4. Project each trip to a local metre-scale coordinate frame before distance and segment calculations. Exclude arcs with invalid node references and zero geometric length from candidate geometry.
5. Generate 12 nearby road-segment candidates per observation from segment midpoints using a spatial index, then calculate exact point-to-segment features.
6. Build training labels by matching each GPS observation to the closest road segment in that trip’s supplied reference route. Candidate-miss observations are measured and omitted from binary candidate-ranker fitting because the correct segment is absent from their candidate list.

The supplied dataset audit found 0 invalid-coordinate rows and 0 non-increasing timestamps in the inspected partitions. So the cleaning rules ran, but did not remove rows from these files. The 1-second sampling pattern is nominal; gaps over 5 seconds are retained and reported.

## Feature choices and reasons

| Feature | What it describes | Why use it for road matching |
|---|---|---|
| Distance to segment (m) | Perpendicular distance from GPS point to candidate road segment | Noisy GPS is usually closest to the traveled road; metre units make distance interpretable. |
| Heading difference (radians) | Difference between the vehicle’s recent direction and candidate segment direction | Helps distinguish nearby parallel or oppositely directed roads. |
| Along-segment fraction | Projected position between the segment endpoints, clipped to 0–1 | Indicates whether the point lies near the candidate segment’s span versus near its extension. |
| Segment length (m) | Length of the candidate arc | Provides local geometry context for interpreting distance and endpoint position. |
| GPS speed (m/s) | Distance moved since the previous observation divided by elapsed seconds | Adds movement context and respects irregular time gaps. |

This is a compact, task-motivated phase-1 feature set. A formal feature-ablation study has not yet been run, so the team should not claim these are globally optimal. Logistic Regression and KNN use standardized features; tree models use the metre/radian feature values directly.

## Five-model comparison

Models were fitted on candidate rows from 64 training trips; 16 whole training trips were held out for phase-1 validation. All five learned models used the same candidate features and labels. Student-test results are shown as development-stage comparisons only. The candidate-miss rate is shared by models because candidate generation is shared.

| Model | Validation Top-1 | Validation candidate Macro F1 | Student-test Top-1 | Student-test candidate Macro F1 | Student-test log loss | Inference seconds (student test) |
|---|---:|---:|---:|---:|---:|---:|
| Nearest road segment | 55.37% | — | 53.48% | — | — | — |
| Logistic Regression | 55.22% | 0.6301 | 56.50% | 0.6022 | 0.4722 | 0.0047 |
| KNN | 60.79% | 0.6965 | 65.69% | 0.7251 | 0.5413 | 0.4709 |
| Decision Tree | 68.87% | 0.7989 | 72.94% | 0.8024 | 0.2230 | 0.0076 |
| Random Forest | 71.09% | 0.8323 | 75.14% | 0.8414 | 0.1385 | 0.1834 |
| Hist Gradient Boosting | 71.45% | 0.8478 | 75.45% | 0.8609 | 0.1065 | 0.1146 |

### Recorded initial configurations

| Model | Phase-1 configuration |
|---|---|
| Logistic Regression | StandardScaler; max_iter=250; class_weight=balanced |
| KNN | StandardScaler; k=9; distance-weighted neighbors; 5,000 balanced candidate rows for practical inference |
| Decision Tree | max_depth=14; min_samples_leaf=4; class_weight=balanced |
| Random Forest | 80 trees; max_depth=18; min_samples_leaf=2; balanced_subsample |
| Hist Gradient Boosting | 100 boosting iterations; learning_rate=0.08; max_leaf_nodes=15; L2=1.0 |

### Initial model choice

**Hist Gradient Boosting** is the provisional phase-1 selection because it had the highest whole-trip validation Top-1 accuracy (71.45%), narrowly above Random Forest (71.09%). It also had lower measured student-test inference time than Random Forest (0.115s vs 0.183s) and the highest student-test Top-1 in this initial comparison (75.45%).
The validation margin over Random Forest is small (about 0.36 percentage points), so this is an initial choice, not a final claim of superiority. A later phase should use group-aware cross-validation, hyperparameter search, probability calibration, and a candidate-generation ablation before finalizing the model.

Candidate Macro F1 and log loss here are computed at the binary candidate-row level (correct candidate vs other candidate), not as a globally shared road-ID classification score. Arc IDs are local to each trip.

## Initial HMM/Viterbi comparison

On all 10 student-test trips (19,160 observations), classifier Top-1 was 75.45% and the initial HMM/Viterbi output was 57.64%. The strict direct-arc continuity rate changed from 25.41% to 73.86% (trip-average).

This is a visible working decoder, but its first transition rule trades pointwise accuracy for direct adjacency. Treat it as an initial prototype and a clear next improvement: allow plausible multi-arc paths between GPS samples, account for elapsed time and speed, and tune decoder parameters on permitted development data. Do not describe the current decoder as improving accuracy.

## Reproducible demonstration

The local web app lets a reviewer select a student-test trip, runs candidate generation → selected classifier → Viterbi decoding, and displays the predicted route, metrics, and an interactive Folium map with GPS points, reference route, candidate road context, and predicted route. The app and API run on one laptop.

## Phase-1 limits

- Evaluator-test data was not included in the supplied ZIP and was not used.
- Initial validation is a single trip-wise holdout rather than the full group-aware cross-validation requested for the complete submission.
- The route label for each observation is approximated by nearest geometry on the supplied route sequence; candidate misses remain a measurable ceiling.
- The Viterbi transition model is an initial implementation; the first all-test comparison shows an accuracy/continuity tradeoff that needs improvement.
- Feature ablations, Bayesian optimization, full final report, and the final evaluator-facing trip flow remain later work.
