"""Single-laptop evaluation API. Evaluator route labels stay server-side."""
from pathlib import Path
import os, json
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .data import Trip, read_trip
from .matcher import generate_candidates,predict_batch,viterbi_decode,reference_labels
from .visualize import create_map

ROOT=Path(os.environ.get("PATHSNAP_ROOT",Path(__file__).resolve().parents[2]))
DEV_DATA=Path(os.environ.get("PATHSNAP_DATA",ROOT/"data"/"Student folder"))
EVALUATOR_DATA=Path(os.environ.get("PATHSNAP_EVALUATOR_DATA",ROOT/"data"/"evaluator_test"))
MODEL_PATH=Path(os.environ.get("PATHSNAP_MODEL",ROOT/"outputs"/"phase2_model.joblib"))
app=FastAPI(title="PathSnap Evaluation",version="2.0.0")
# The separate UI laptop calls the API across the local network. No credentials are used.
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["GET","POST","OPTIONS"],allow_headers=["*"])
app.mount("/maps",StaticFiles(directory=ROOT/"maps",check_dir=False),name="maps")

@app.get("/")
def home(): return FileResponse(ROOT/"web"/"index.html")

@app.get("/api/health")
def health():
    return {"status":"ready","model_exists":MODEL_PATH.exists(),"evaluator_trip_count":len(_trip_ids(EVALUATOR_DATA))}

def _trip_ids(root: Path):
    if not root.exists(): return []
    return [p.name for p in sorted(root.iterdir()) if p.is_dir() and (p/f"{p.name}.track").exists()]

@app.get("/api/trips")
def trips():
    """List evaluator IDs only; no route labels are returned."""
    return {"trips":_trip_ids(EVALUATOR_DATA)}

@app.get("/api/dev/trips")
def dev_trips():
    """Development-only student-test list for local debugging."""
    return {"trips":_trip_ids(DEV_DATA/"student_test")}

def _predict_from_partition(root: Path, trip_id: str, is_evaluator: bool):
    if not MODEL_PATH.exists(): raise HTTPException(503,"Train the notebooks/06_Phase2_Model_Validation_Tuning.ipynb first")
    if trip_id not in _trip_ids(root): raise HTTPException(404,"Trip ID is not in this evaluation partition")
    trip_path=root/trip_id
    # The evaluator reference route is loaded only by this server-side evaluator.
    labeled_trip=read_trip(trip_path,"evaluator_test" if is_evaluator else "student_test")
    # Candidate generation/model receive a copy with no reference labels.
    inference_trip=Trip(labeled_trip.trip_id,labeled_trip.partition,labeled_trip.track,labeled_trip.nodes,labeled_trip.arcs,np.empty(0,dtype=np.int64),labeled_trip.raw_track_rows)
    bundle=joblib.load(MODEL_PATH)
    batch=generate_candidates(inference_trip,bundle["candidate_k"])
    classifier,probs=predict_batch(bundle["model"],batch)
    gaps=np.diff(inference_trip.track[:,2],prepend=inference_trip.track[0,2])
    decoded=viterbi_decode(batch.ids,probs,inference_trip.arcs,gaps)

    # Only the server-side evaluation step reads reference labels. They are never
    # passed to candidate generation, model prediction, the map, or the UI.
    truth=reference_labels(labeled_trip)
    valid=truth>=0
    candidate_miss=~np.any(batch.ids==truth[:,None],axis=1)
    def continuity(ids):
        if len(ids)<2:return 0.0
        return float(np.mean(inference_trip.arcs[ids[:-1],1]==inference_trip.arcs[ids[1:],0]))
    metrics={"trip_id":trip_id,"model":bundle["model_name"],"observations":int(valid.sum()),
        "candidate_miss_rate":float(candidate_miss[valid].mean()) if valid.any() else None,
        "classifier_top1_accuracy":float((classifier[valid]==truth[valid]).mean()) if valid.any() else None,
        "viterbi_top1_accuracy":float((decoded[valid]==truth[valid]).mean()) if valid.any() else None,
        "classifier_direct_arc_continuity":continuity(classifier),"viterbi_direct_arc_continuity":continuity(decoded),
        "reference_labels_hidden":True}
    prefix="evaluator" if is_evaluator else "dev"
    safe_name=f"{prefix}_{trip_id}"
    map_path=ROOT/"maps"/f"{safe_name}.html"
    create_map(inference_trip,batch,decoded,map_path,show_reference_route=False)
    # The optional server-side record contains metrics only, never reference arcs.
    (ROOT/"outputs"/f"{safe_name}.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    return {**metrics,"map_url":f"/maps/{safe_name}.html","predicted_arc_ids":decoded.tolist()}

@app.post("/api/predict/{trip_id}")
def predict(trip_id: str):
    return _predict_from_partition(EVALUATOR_DATA,trip_id,True)

@app.post("/api/dev/predict/{trip_id}")
def predict_development(trip_id: str):
    return _predict_from_partition(DEV_DATA/"student_test",trip_id,False)
