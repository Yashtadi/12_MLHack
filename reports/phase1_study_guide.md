# Round 1 presenter walkthrough

## 60-second problem explanation

GPS is noisy, so recorded points can fall beside the road the vehicle actually used. PathSnap receives GPS points and a trip-specific road graph. It proposes nearby road segments for each point, scores them using geometry and movement, then uses the sequence and road connectivity to form a route.

The four files are `.track` (GPS longitude, latitude, time), `.nodes` (road coordinates), `.arcs` (directed connections between node rows), and `.route` (reference arc-row IDs in travel order). Arc IDs are local to a trip.

## Walk the notebooks in order

1. **Dataset exploration:** show the inventory of 80 training and 10 student-test trips, the four schemas, GPS quality checks, timestamp gaps, trip-size summaries, and the sample map. Mention that the evaluator partition was not supplied.
2. **Preprocessing:** show invalid-value checks, timestamp handling, how gaps are preserved, coordinate projection to metres, graph validity checks, and the resulting clean arrays. The quality checks found no invalid coordinates or non-increasing timestamps, so this run did not drop rows.
3. **Candidates and features:** explain the KD-tree shortlist of 12 nearby segment midpoints. Show the visible worked formula: project the GPS point onto each segment, clip the fraction to [0,1], then measure distance. Explain the five features: distance, heading difference, along-segment fraction, segment length, and GPS speed. The first three ranked highest in the preliminary permutation-importance check. Candidate-miss rate matters because a model cannot select a road that candidate generation omitted.
4. **Model training:** show the whole-trip split (64 fit trips, 16 validation trips), balanced candidate sampling, each model's setup, fit loop, validation metrics, and winner-selection rule. This is actual training code and saved model output, not a report-only notebook.
5. **Evaluation:** show the comparison table, preliminary feature importance, refit model artifact, student-test results, and Folium route map.

## Explain why this model was chosen

The five trained methods are Logistic Regression, KNN, Decision Tree, Random Forest, and Hist Gradient Boosting. We choose using validation Top-1 on held-out trips, not by the student-test scores. Hist Gradient Boosting reached 71.45%, narrowly ahead of Random Forest at 71.09%, so it is our provisional Round 1 model. We refit it on all 80 training trips after choosing it. The lead is only 0.36 percentage points; one split is not enough to claim it is definitively best.

Top-1 means the highest-ranked candidate matches the approximate reference label at an observation. Candidate Macro F1 and log loss treat each candidate as correct or alternative, since road IDs cannot be shared across trip-specific graphs.

## Explain the Viterbi result honestly

Across the 10 student-test trips, the classifier had 75.45% Top-1 and the initial Viterbi decoder had 57.64%. Direct arc-to-arc continuity increased from 25.41% to 73.86%. The decoder trades pointwise accuracy for locally connected routes. Describe it as a working prototype, not an accuracy improvement. The next step is to score plausible multi-arc paths using time and distance.

## Show the live demo

Start the app from the project folder with:

```bash
uvicorn pathsnap.server:app --app-dir src --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`, choose a student-test trip, and click **Predict route**. Explain the flow: GPS and graph → candidate arcs and features → saved Hist Gradient Boosting scores → Viterbi path → Folium map. The reference route is used for development metrics.

## Key limitations to disclose

The evaluator-only partition was unavailable. Validation is one trip-wise holdout; feature selection is preliminary and needs ablation; candidate labels use nearest route geometry; and the Viterbi step improves direct continuity while lowering Top-1. The results are initial development evidence, not final evaluator performance.
