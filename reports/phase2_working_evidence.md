# PathSnap Phase 2 — Working Evidence

## Dataset partitions and integrity

The supplied student data contains 80 training trips and 10 student-test trips. The evaluator ZIP contains a separate set of 10 trips and is stored locally under the ignored `data/evaluator_test/` directory. It is not included in the model-training or Bayesian-optimization notebook. The evaluator routes are used only by the API's final server-side metric calculation; the prediction copy of each trip has an empty route array.

## Feature set carried forward from Phase 1

The Phase 2 notebook reuses the five features implemented in `src/pathsnap/matcher.py`: perpendicular GPS-to-segment distance (`distance_m`), absolute GPS movement-versus-road heading difference (`heading_difference_rad`), clipped position along the segment (`along_segment_fraction`), road segment length (`segment_length_m`), and GPS speed since the previous observation (`gps_speed_mps`). The Phase 2 report describes this same feature set. The feature-importance check in Phase 1 ranked the first three highest; all five were retained for the comparisons.

## Phase 2 model comparison

The 80 training trips were divided into five GroupKFold folds. All rows from a trip remain in one fold. Candidate-level positive rows match the nearest geometry on the supplied reference route; alternative candidates are negative. Candidate misses are excluded from fitting. Each fold uses all positive fit rows and up to four sampled negatives per positive, with the fixed seed 42 and a 200,000-row cap. KNN is additionally limited to a balanced 5,000-row fit sample.

| Model | CV candidate Macro F1 (mean ± SD) | CV Top-1 mean | Student Top-1 | Student candidate Macro F1 | Student log loss |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6358 ± 0.0382 | 50.23% | 57.80% | 0.6088 | 0.4637 |
| KNN | 0.7202 ± 0.0188 | 59.53% | 66.11% | 0.7233 | 0.5437 |
| Decision Tree | 0.7981 ± 0.0143 | 65.76% | 73.32% | 0.7980 | 0.2287 |
| Random Forest | 0.8327 ± 0.0183 | **68.47%** | 75.23% | 0.8413 | 0.1400 |
| Hist Gradient Boosting | **0.8485 ± 0.0251** | 67.87% | **75.44%** | **0.8613** | **0.1067** |

Candidate Macro F1 is calculated across binary candidate rows (correct candidate vs. alternative). Top-1 selects the highest-scoring candidate for each GPS observation and compares it with the approximate reference label. These are different metrics: Random Forest had higher average cross-validation Top-1, while Hist Gradient Boosting had higher mean cross-validation candidate Macro F1. We selected the model family using the specified CV Macro F1 objective. The student-test scores are development comparisons, not the selection criterion.

Student candidate precision and recall for untuned Hist Gradient Boosting were 0.6547 and 0.8720 respectively. The student candidate miss rate was 6.35% for every model because candidate generation was shared.

## Bayesian hyperparameter search

The selected Hist Gradient Boosting family was tuned using 15 Optuna TPE trials. The objective was mean candidate Macro F1 over the same five grouped training folds. Student-test and evaluator data were not supplied to this objective.

Best trial settings:

- `max_iter`: 60
- `learning_rate`: 0.0259
- `max_leaf_nodes`: 31
- `min_samples_leaf`: 47
- `l2_regularization`: 0.0487

The mean CV candidate Macro F1 increased from 0.8485 to 0.8516. The fold standard deviation changed from 0.0251 to 0.0279, so the improvement is modest and fold variability remains. On student-test, the tuned model had Top-1 75.40% versus 75.44% untuned; candidate Macro F1 0.8667 versus 0.8613; precision 0.6914 versus 0.6547; recall 0.8348 versus 0.8720; and log loss 0.1352 versus 0.1067. Tuning improved the CV objective and student candidate Macro F1, while student Top-1 and probability log loss were slightly worse. We retained the tuned model because the declared selection objective was CV candidate Macro F1, and we disclose the tradeoff.

The tuned model bundle is `outputs/phase2_model.joblib`. Trial results, fold details, and comparison tables are in `outputs/phase2_bayesian_trials.csv`, `outputs/phase2_cv_folds.csv`, `outputs/phase2_cv_model_comparison.csv`, `outputs/phase2_student_comparison.csv`, and `outputs/phase2_tuned_comparison.csv`.

## HMM/Viterbi route reconstruction

Each hidden state is one of the 12 candidate road arcs for an observation. Emissions are the selected classifier's candidate probabilities. Transitions reward directed arc connectivity, mildly penalize staying on the same arc, and strongly penalize disconnected candidates. For GPS gaps over five seconds, the disconnected transition penalty is relaxed. Dynamic programming stores the best cumulative score and backpointers, then reconstructs the highest-scoring candidate sequence.

The Phase 1 student-test comparison showed that this direct-adjacency decoder raised strict direct-arc continuity from 25.41% to 73.86% but lowered pointwise Top-1 from 75.45% to 57.64%. With the Phase 2 tuned classifier on all 10 student-test trips (19,160 observations), classifier Top-1 was 75.40% and Viterbi Top-1 was 56.10%; trip-mean direct continuity moved from 25.69% to 77.80%. It is a transparent initial decoder with a real accuracy/continuity tradeoff. It does not yet model shortest multi-arc paths between samples, so route coherence and point accuracy should be improved in a later iteration using permitted development data.

## Evaluator API privacy

The evaluator-facing `GET /api/trips` lists trip IDs only. For inference, the server constructs a separate `Trip` whose route array is empty and passes that to candidate generation, the classifier, Viterbi, and map rendering. Only after predictions are produced does a server-side evaluator associate observations with the reference route and compute metrics. The API returns aggregate metrics and predicted arc IDs; the map includes observed GPS, candidate road context, and predicted route, with no reference-route layer.

The `data/evaluator_test/` directory and per-trip evaluator output files are ignored by Git. No evaluator labels are used for model fitting, feature selection, hyperparameter search, decoder tuning, or threshold selection.

## Limits and next improvements

- Route labels are approximate: each GPS point is assigned the geometrically nearest arc among the supplied route, which may be ambiguous near crossings or loops.
- Candidate generation uses a 12-midpoint shortlist; its miss rate limits achievable point accuracy.
- Candidate Macro F1 is not the same as route-level Macro F1; the report labels it explicitly.
- Bayesian search used 15 trials and a single five-fold grouped CV design; broader search or nested CV could provide more robust selection.
- Viterbi currently uses direct adjacency rather than realistic multi-arc shortest paths.
- The evaluator partition is reserved for one final server-side check and must not influence model or decoder changes.

## Final evaluator run after model freeze

The frozen tuned Hist Gradient Boosting model and the unchanged Viterbi decoder were sent through the evaluator API for all 10 evaluator trips (19,674 observations). The API computed these scores server-side after producing predictions; evaluator labels were not used to revise features, hyperparameters, thresholds, model choice, or decoder settings.

| Metric | Classifier | After Viterbi |
|---|---:|---:|
| Observation-level Top-1 | **69.49%** | 54.36% |
| Direct-arc continuity (trip mean) | 30.12% | **81.40%** |

The evaluator candidate-miss rate was **4.79%**. As on student-test, the initial decoder improves direct continuity but reduces pointwise accuracy. These are the final one-time evaluator results and were not fed back into development. The reference-route association uses the same nearest-geometry approximation described above.
