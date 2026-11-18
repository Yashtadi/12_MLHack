# PathSnap — Round 1 demo

PathSnap matches noisy GPS observations to a trip-specific road network. The demo proposes nearby road arcs, scores them with a trained model, applies an initial Viterbi decoder, and displays the result on a Folium map. Everything runs on one laptop.

## Demo contents

Open the executed notebooks in order. They include saved outputs from the supplied data and show the actual calculations and decisions.

1. `notebooks/01_Dataset_Exploration.ipynb` — inspect the four file formats, dataset sizes, quality, time gaps, and a sample trip.
2. `notebooks/02_Data_Preprocessing.ipynb` — validate and clean GPS rows, preserve time gaps, and prepare road geometry.
3. `notebooks/03_Candidate_Generation_Features.ipynb` — generate candidate arcs, calculate and inspect five features, and measure candidate coverage.
4. `notebooks/04_ML_Model_Training.ipynb` — create a trip-wise holdout, fit five classifiers, compare validation results, choose a model, and refit it.
5. `notebooks/05_Initial_Evaluation.ipynb` — evaluate the chosen model and initial route decoder, inspect feature importance, and create a map.

`reports/phase1_study_guide.md` is the presenter walkthrough. `reports/phase1_working_evidence.md` records measured results and limitations.

## Data

The raw dataset is intentionally excluded from Git. Extract the supplied `Student folder` under `data/`, so these folders exist:

```text
data/Student folder/train/<trip_id>/
data/Student folder/student_test/<trip_id>/
```

Each trip folder contains `.track`, `.nodes`, `.arcs`, and `.route` files. The supplied archive has 80 training trips and 10 student-test trips; evaluator data was not included or used.

## Install and run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Open the notebooks in VS Code or Jupyter, select the `.venv` Python kernel, and run from top to bottom. Saved outputs are included, so the presentation can begin without rerunning training. Notebook 04 reruns all five models and saves `outputs/phase1_model.joblib`; notebooks 01 and 05 refresh the inventory and evaluation evidence.

To start the local browser demo after the model artifact exists:

```bash
uvicorn pathsnap.server:app --app-dir src --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`, select a student-test trip, and click **Predict route**. The app runs candidate generation, the saved classifier, Viterbi decoding, and map creation locally.

## Modeling notes

Road arc IDs are local to each trip, so the classifier uses geometry and movement features instead of treating IDs as shared classes. Reference route labels are approximated by assigning each GPS point to its nearest segment in the supplied route. Candidate misses are measured because the classifier cannot select a road absent from its candidate list.

Hist Gradient Boosting is the provisional choice based on the highest Top-1 accuracy on a whole-trip validation holdout. Its lead over Random Forest is small, so the evidence supports an initial choice, not a claim of final superiority. The initial Viterbi decoder improves strict direct-arc continuity but lowers pointwise accuracy; that tradeoff is shown in the evaluation notebook.
