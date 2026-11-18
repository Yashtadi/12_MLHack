# Phase 1 study guide and demo script

## 60-second problem explanation

GPS points can drift away from the road because of noise, buildings, and sampling gaps. PathSnap takes a sequence of GPS observations and the trip’s road graph, proposes nearby road segments for each observation, scores those candidates, then uses the order of observations and road connectivity to reconstruct a more coherent route.

The four files are `.track` (longitude, latitude, time), `.nodes` (road-node coordinates), `.arcs` (directed connections between node rows), and `.route` (reference arc-row IDs in travel order).

## Explain the preprocessing

1. Read each headerless file with its known column count and keep trips grouped by their directory.
2. Remove GPS rows only when coordinates/timestamps are invalid, coordinates exceed valid longitude/latitude ranges, or time does not increase. Preserve row order; do not shuffle a trajectory.
3. The supplied partitions had zero invalid coordinates and zero backwards/duplicate timestamps, so these rules removed no rows in this run.
4. Keep large time gaps and compute speed using the real timestamp delta; a long gap must not be treated as one second.
5. Convert coordinates to a local metre-scale plane before distance calculations. Degrees are angular units and are unsuitable for direct physical-distance comparisons.
6. Exclude road arcs with invalid node references or zero length from candidate geometry.
7. For each GPS point, retrieve 12 segment midpoints from a KD-tree, then calculate the exact nearest point on each segment. Measure how often the reference road is missing from those candidates.

## Explain the five features

- **Distance to segment:** closest physical separation in metres; usually a strong cue for which road generated the GPS point.
- **Heading difference:** compares recent vehicle motion with the candidate road direction; helps separate nearby parallel or opposite-direction roads.
- **Along-segment fraction:** tells whether the GPS point projects near the segment’s span or beyond its endpoint.
- **Segment length:** gives context about the candidate edge geometry.
- **Speed:** adds movement context and accounts for irregular time gaps.

This is a compact, physically motivated first set, not a claim that formal feature selection is complete. Feature ablation is still needed.

## Explain model choice with the measured results

Five models were trained with the same candidate features: Logistic Regression, KNN, Decision Tree, Random Forest, and Hist Gradient Boosting. Whole trips were held out from the 80 training trips, so points from one trip did not leak into both fit and validation.

Hist Gradient Boosting was selected provisionally because it had the best holdout Top-1 accuracy (**71.45%**) versus Random Forest (**71.09%**), and slightly higher student-test Top-1 (**75.45%** vs **75.14%**) with faster measured inference in this run (**0.108 s** vs **0.192 s**). The holdout gap is only about 0.36 percentage points, so explain that this is an early choice that needs group cross-validation and tuning.

Top-1 accuracy means the highest-scored candidate matches the reference road for that GPS observation. Candidate Macro F1 and log loss are calculated over binary candidate rows (correct candidate versus other candidate), because local arc IDs are not shared across trips. Candidate miss rate is the fraction of observations whose reference road was absent from the 12 candidates; no classifier can recover a segment it was never offered.

## Explain the HMM/Viterbi demo honestly

The initial decoder favors adjacent directed arcs. Across the 10 student-test trips, strict direct-arc continuity rose from **25.41%** to **73.86%**, while Top-1 accuracy fell from **75.45%** to **57.64%**. This shows the tradeoff: stronger connectivity can make a sequence look more coherent but can override locally correct classifications. The next improvement is to allow realistic multi-arc paths between samples, use time and distance to score transitions, and tune only on permitted development partitions.

## Click-through demonstration

1. From `ML_hackathon`, start the app with `uvicorn pathsnap.server:app --app-dir src --host 127.0.0.1 --port 8000`.
2. Open `http://127.0.0.1:8000` and select a student-test trip.
3. Click **Predict route**. Explain that the API uses GPS, the road graph, the candidate features, and the saved Hist Gradient Boosting model; the reference route is used only to calculate development metrics.
4. Show the map layers for observed GPS, candidate road context, reference route, and predicted route.
5. Show `reports/phase1_working_evidence.md`, `outputs/model_comparison.csv`, and `outputs/route_decoder_comparison.csv` as the evidence behind the choices.

## Be clear about what is unfinished

The evaluator-only partition was not in the supplied ZIP. Bayesian optimization, group-aware cross-validation, feature ablations, a stronger route-transition model, and final evaluator integration are future steps. Do not describe the initial phase-1 metrics as final evaluation results.
