# PathSnap — Round 1 and Phase 2

PathSnap matches noisy GPS observations to a trip-specific road network. It generates nearby road candidates, scores them with a trained classifier, applies a Viterbi route decoder, and displays the predicted route on a Folium map. The system supports both a single-laptop demo and the required two-laptop final evaluation.

For a beginner-friendly explanation of what we built after Phase 1, the exact features and metrics, and a viva speaking summary, read [`reports/phase1_to_phase2_beginner_guide.md`](reports/phase1_to_phase2_beginner_guide.md).

## Notebooks

Open these in order for the data-to-prediction walkthrough:

1. `notebooks/01_Dataset_Exploration.ipynb` — inspect file formats, partitions, quality, time gaps, and sample trips.
2. `notebooks/02_Data_Preprocessing.ipynb` — validate GPS, preserve timestamps and gaps, inspect graph references, project coordinates.
3. `notebooks/03_Candidate_Generation_Features.ipynb` — generate 12 road candidates and calculate the five geometry/movement features.
4. `notebooks/04_ML_Model_Training.ipynb` — Phase 1 five-model baseline and trip-wise holdout.
5. `notebooks/05_Initial_Evaluation.ipynb` — Phase 1 comparison, initial decoder analysis, and map.
6. `notebooks/06_Phase2_Model_Validation_Tuning.ipynb` — five-fold GroupKFold comparison, student development metrics, 15-trial Optuna TPE search, tuned/untuned comparison, and final model fitting. Evaluator trips are not loaded in this notebook.

Executed notebook outputs and comparison CSVs are included. `reports/phase1_study_guide.md` and `reports/phase2_working_evidence.md` record the earlier walkthrough and measured results. `reports/PathSnap_Phase1_Phase2_Technical_Report.pdf` is the single combined technical report; rebuild it with `.venv/bin/python reports/build_technical_report.py`.

## Data setup

Training and student-test data belong under `data/Student folder/`. Extract the evaluator ZIP on Laptop 1 under `data/evaluator_test/`. These data folders and per-trip evaluator outputs are ignored by Git; do not copy evaluator data to Laptop 2.

```text
data/Student folder/train/<trip_id>/
data/Student folder/student_test/<trip_id>/
data/evaluator_test/<trip_id>/
```

Each trip directory contains `.track`, `.nodes`, `.arcs`, and `.route`. During inference, the model receives a trip copy with the reference route removed. The API uses its server-side route only after prediction to calculate evaluation metrics.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate                 # Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```

The saved final model is `outputs/phase2_model.joblib`. Run notebook 06 in the `.venv` kernel only if you need to retrain it.

## Two-laptop final evaluation

Both laptops must be on the same trusted Wi-Fi/LAN. Laptop 1 keeps the model and evaluator data; Laptop 2 only serves the browser UI. If prompted, allow Python through Laptop 1's firewall on the private network. Do not use a guest network that blocks devices from connecting to one another.

### Laptop 1 — API, model and evaluator data

Open a terminal in the project root, activate `.venv`, and run:

```bash
PATHSNAP_EVALUATOR_DATA='data/evaluator_test' python -m uvicorn pathsnap.server:app --app-dir src --host 0.0.0.0 --port 8000
```

On Windows PowerShell, set the data path and start the same API with:

```powershell
$env:PATHSNAP_EVALUATOR_DATA = "data/evaluator_test"
python -m uvicorn pathsnap.server:app --app-dir src --host 0.0.0.0 --port 8000
```

Leave this terminal open. On macOS, find Laptop 1's Wi-Fi IP with `ipconfig getifaddr en0`. On Windows, run `ipconfig` and find the IPv4 address under the Wi-Fi adapter. Verify the API on Laptop 1 at `http://127.0.0.1:8000/api/health`; it should report `model_exists: true` and 10 evaluator trips.

### Laptop 2 — browser UI

Clone the public repo if needed, then open a terminal in its `web/` folder and run:

```bash
python3 -m http.server 8080 --bind 127.0.0.1
```

On Windows, use `py -m http.server 8080 --bind 127.0.0.1`. Leave the terminal open and browse to `http://127.0.0.1:8080` on Laptop 2. In **Laptop 1 API address**, enter `http://<LAPTOP-1-IP>:8000` (for example, `http://192.168.1.25:8000`) and click **Connect**. Confirm the UI says it is connected and lists 10 trips.

During evaluation, let the evaluator select a trip and click **Evaluate trip**. Laptop 2 sends the trip ID. Laptop 1 runs the prediction and server-side metric calculation, and its terminal shows the `/api/predict/<trip_id>` request. The UI displays the predicted route, map, and evaluation results; raw reference route IDs stay on Laptop 1 and are not sent to the model or UI.

For a one-laptop demo, start the same API with `--host 127.0.0.1 --port 8000` and open `http://127.0.0.1:8000`.

## Model selection summary

The five-model comparison uses five whole-trip GroupKFold folds over the 80 training trips. Candidate Macro F1 is the selection objective. Hist Gradient Boosting led that CV objective and was tuned with Optuna TPE using training folds only. Student-test results are development comparisons. Evaluator results are calculated by the API after the model is frozen. See the beginner guide and Phase 2 evidence report for measured values and tradeoffs.
