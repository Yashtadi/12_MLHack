# PathSnap — Phase 1 working prototype

A single-laptop demonstration of GPS map matching: inspect trip files, generate nearby road candidates, rank them with a learned model, decode a route with Viterbi, and view the result on a Folium map.

## Phase 1 status

- Dataset exploration and preprocessing are implemented for `.track`, `.nodes`, `.arcs`, and `.route` files.
- Candidate generation uses a spatial index and exact point-to-segment distances.
- Five classifiers are trained and compared using a whole-trip holdout from the 80 training trips.
- Hist Gradient Boosting is the provisional model selected from holdout Top-1 performance.
- Initial HMM/Viterbi route decoding, Folium maps, and a local browser interface are available.
- The provided archive contains 80 training trips and 10 student-test trips. It does not contain the evaluator-test partition, which has not been used.

Open the five executed notebooks in order under `notebooks/`: `01_Dataset_Exploration.ipynb`, `02_Data_Preprocessing.ipynb`, `03_Candidate_Generation_Features.ipynb`, `04_ML_Model_Training.ipynb`, and `05_Initial_Evaluation.ipynb`. Each contains saved outputs from the actual phase-1 data and training run. See [`reports/phase1_working_evidence.md`](reports/phase1_working_evidence.md) for the measured summary and [`reports/phase1_study_guide.md`](reports/phase1_study_guide.md) for a short presenter walkthrough.

## Data setup

The raw data is intentionally not tracked in Git. Put the extracted `Student folder` directory under `data/` so that the paths look like:

```text
data/Student folder/train/<trip_id>/<trip_id>.track
 data/Student folder/student_test/<trip_id>/<trip_id>.track
```

Each trip directory should also contain its `.nodes`, `.arcs`, and `.route` files. Evaluator-only data should remain separate and must not be used for development or model selection.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Open the Round 1 notebooks

Install the requirements, open this folder in VS Code/Jupyter, select the `.venv` Python kernel, and open the five notebooks in numeric order. The fourth notebook reruns all five models; the others show preprocessing, candidate features, and the actual measured results. Saved outputs are already included for quick presentation.

## Run phase-1 exploration and training

From the project directory:

```bash
python src/eda_report.py --data-dir 'data/Student folder' --output-dir results
python src/train_phase1.py --data-dir 'data/Student folder' --output-dir outputs --candidates 12
python src/evaluate_routes.py --data-dir 'data/Student folder' --model outputs/phase1_model.joblib --out outputs/route_decoder_comparison.csv
python src/build_phase1_evidence.py .
python src/create_phase1_demo.py --data-dir 'data/Student folder' --model outputs/phase1_model.joblib --trip-id 00000000 --output maps/phase1_demo.html
```

Generated evidence includes a trip inventory, EDA plots, a five-model comparison table, selected model artifact, route-decoder comparison, and interactive Folium map.

## Start the one-laptop app

```bash
uvicorn pathsnap.server:app --app-dir src --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Choose a student-test trip and click **Predict route**. The local API generates road candidates, applies the saved classifier and Viterbi decoder, writes an interactive map, and displays development metrics. The API rejects IDs that are not in the student-test partition. The evaluator-only partition is not present in this prototype.

## Modeling notes

Each trip has a separate road network, so arc IDs are local to trips. The model ranks candidate segments using five geometry/movement features rather than treating arc IDs as one global class vocabulary. The supplied route sequence provides training reference labels by assigning each GPS observation to the closest route segment. Candidate-miss observations are measured and excluded from candidate-ranker fitting.

The phase-1 decoder improves strict direct-arc continuity on the student-test set, but lowers pointwise Top-1 accuracy. Its transition rule is therefore an initial prototype that needs multi-arc path costs and further development before final use.

## Integrity and reproducibility

The report contains actual outputs from the scripts above. Review and understand every result before presenting it; disclose AI assistance according to course policy. Do not use evaluator labels for model selection, tune against hardcoded trip answers, or present the phase-1 results as final evaluator performance.
