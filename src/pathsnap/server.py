"""Single-laptop FastAPI demo. Student-test labels are used only for development display."""
from pathlib import Path
import os, json
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .data import read_trip
from .matcher import generate_candidates,predict_batch,viterbi_decode
from .visualize import create_map

ROOT=Path(os.environ.get("PATHSNAP_ROOT",Path(__file__).resolve().parents[2]))
DATA=Path(os.environ.get("PATHSNAP_DATA",ROOT/"data"/"Student folder"))
MODEL_PATH=Path(os.environ.get("PATHSNAP_MODEL",ROOT/"outputs"/"phase1_model.joblib"))
app=FastAPI(title="PathSnap Phase 1",version="0.1.0")
app.mount("/maps",StaticFiles(directory=ROOT/"maps",check_dir=False),name="maps")

@app.get("/")
def home(): return FileResponse(ROOT/"web"/"index.html")

@app.get("/api/trips")
def trips():
    root=DATA/"student_test"
    if not root.exists(): return {"trips":[]}
    return {"trips":[p.name for p in sorted(root.iterdir()) if p.is_dir() and (p/f"{p.name}.track").exists()]}

@app.post("/api/predict/{trip_id}")
def predict(trip_id: str):
    if not MODEL_PATH.exists(): raise HTTPException(503,"Run all cells in notebooks/04_ML_Model_Training.ipynb to train the phase-1 model first")
    trip_path=DATA/"student_test"/trip_id
    if not (trip_path/f"{trip_id}.track").exists(): raise HTTPException(404,"Trip is not in the development student-test partition")
    bundle=joblib.load(MODEL_PATH); trip=read_trip(trip_path,"student_test")
    batch=generate_candidates(trip,bundle["candidate_k"]); classifier,probs=predict_batch(bundle["model"],batch)
    gaps=np.diff(trip.track[:,2],prepend=trip.track[0,2]); decoded=viterbi_decode(batch.ids,probs,trip.arcs,gaps)
    valid=batch.truth>=0
    safe_name="phase1_"+trip_id
    map_path=ROOT/"maps"/f"{safe_name}.html"
    create_map(trip,batch,decoded,map_path)
    def continuity(ids):
        if len(ids)<2:return 0.0
        return float(np.mean(trip.arcs[ids[:-1],1]==trip.arcs[ids[1:],0]))
    metrics={"trip_id":trip_id,"model":bundle["model_name"],"observations":int(valid.sum()),"candidate_miss_rate":float(batch.candidate_miss[valid].mean()),"classifier_top1_accuracy":float((classifier[valid]==batch.truth[valid]).mean()),"viterbi_top1_accuracy":float((decoded[valid]==batch.truth[valid]).mean()),"classifier_direct_arc_continuity":continuity(classifier),"viterbi_direct_arc_continuity":continuity(decoded)}
    (ROOT/"outputs"/f"{safe_name}.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    return {**metrics,"map_url":f"/maps/{safe_name}.html","predicted_arc_ids":decoded.tolist()}
