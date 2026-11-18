# PathSnap — Round 1 and Phase 2

PathSnap matches noisy GPS observations to the road network for each trip. It generates nearby road candidates, scores them with a trained classifier, applies an initial Viterbi route decoder, and displays the predicted route on a local Folium map. The project runs on one laptop.

## Notebooks

Open these in order for the Phase 1 walkthrough and Phase 2 model selection:

1. `notebooks/01_Dataset_Exploration.ipynb` — inspect file formats, partitions, quality, time gaps, and a sample trip.
2. `notebooks/02_Data_Preprocessing.ipynb` — validate GPS, preserve timestamp order and gaps, inspect graph references, and project coordinates to local metres.
3. `notebooks/03_Candidate_Generation_Features.ipynb` — generate 12 road candidates and calculate five geometry/movement features.
4. `notebooks/04_ML_Model_Training.ipynb` — Phase 1 five-model baseline and trip-wise holdout.
5. `notebooks/05_Initial_Evaluation.ipynb` — Phase 1 comparison, initial Viterbi analysis, and map.
6. `notebooks/06_Phase2_Model_Validation_Tuning.ipynb` — five-fold GroupKFold Macro F1 comparison, student-development metrics, 15-trial Optuna TPE search, tuned/untuned comparison, and final model fitting. This notebook reads only training and student-test data; it never loads evaluator trips.

Saved notebook outputs and CSV evidence are included for presentation. `reports/phase1_study_guide.md` supports the initial walkthrough. `reports/phase2_working_evidence.md` records the Phase 2 method and results.

## Data setup

Training and student-test data belong under `data/Student folder/`. The evaluator ZIP is extracted locally to `data/evaluator_test/`. Both directories are ignored by Git; evaluator routes are never committed.

```text
data/Student folder/train/<trip_id>/
data/Student folder/student_test/<trip_id>/
data/evaluator_test/<trip_id>/
```

Each trip directory contains `.track`, `.nodes`, `.arcs`, and `.route`. The reference route is used by the server-side evaluation step only for evaluator trips. The candidate generator and model receive a copy of the trip with the route labels removed.

## Install and run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Open the notebooks in VS Code or Jupyter and select the `.venv` Python kernel. Notebook 06 saves the final model to `outputs/phase2_model.joblib`.

Start the local evaluation API and interface:

```bash
PATHSNAP_EVALUATOR_DATA='data/evaluator_test' python -m uvicorn pathsnap.server:app --app-dir src --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`, choose an evaluator trip, and click **Evaluate trip**. The server computes accuracy and continuity metrics, while the map shows observed GPS, candidate-road context, and predicted route. The evaluator reference route is not included in the prediction inputs, API response, or map.

The development-only endpoints are `/api/dev/trips` and `/api/dev/predict/{trip_id}`. The evaluator-facing UI uses `/api/trips` and `/api/predict/{trip_id}`.

## Phase 2 model selection

The five-model comparison uses five whole-trip GroupKFold folds over the 80 training trips. Candidate Macro F1 is the model-selection objective. The selected family is then tuned with Optuna's TPE Bayesian sampler using the same grouped folds. Student-test results are reported after selection as development comparisons and are not part of the tuning objective. Evaluator data is kept separate until the final server-side check.

The current model is a provisional experimental result. Hist Gradient Boosting led the five-model comparison in mean cross-validation candidate Macro F1; its single-split Phase 1 Top-1 margin over Random Forest was small. See the Phase 2 evidence report for the tuned-versus-untuned metric tradeoffs and limitations.
