from pathlib import Path
import argparse, json
import numpy as np
import joblib
from pathsnap.data import read_trip
from pathsnap.matcher import generate_candidates,predict_batch,viterbi_decode
from pathsnap.visualize import create_map

if __name__=="__main__":
    ap=argparse.ArgumentParser(description="Create an interactive phase-1 route map for a student-test trip.")
    ap.add_argument("--trip-id",default="00000000"); ap.add_argument("--data-dir",type=Path,default=Path("data/Student folder")); ap.add_argument("--model",type=Path,default=Path("outputs/phase1_model.joblib")); ap.add_argument("--output",type=Path,default=Path("maps/phase1_demo.html"))
    a=ap.parse_args(); bundle=joblib.load(a.model); trip=read_trip(a.data_dir/"student_test"/a.trip_id,"student_test")
    batch=generate_candidates(trip,bundle["candidate_k"]); pred,probs=predict_batch(bundle["model"],batch)
    gaps=__import__('numpy').diff(trip.track[:,2],prepend=trip.track[0,2]); decoded=viterbi_decode(batch.ids,probs,trip.arcs,gaps)
    valid=batch.truth>=0
    def continuity(ids):
        return float(np.mean(trip.arcs[ids[:-1],1]==trip.arcs[ids[1:],0])) if len(ids)>1 else 0.0
    metrics={"trip_id":trip.trip_id,"model":bundle["model_name"],"observations":int(valid.sum()),"candidate_miss_rate":float(batch.candidate_miss[valid].mean()),"classifier_top1_accuracy":float((pred[valid]==batch.truth[valid]).mean()),"viterbi_top1_accuracy":float((decoded[valid]==batch.truth[valid]).mean()),"classifier_direct_arc_continuity":continuity(pred),"viterbi_direct_arc_continuity":continuity(decoded)}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.with_suffix('.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    create_map(trip,batch,decoded,a.output)
    print(json.dumps(metrics,indent=2)); print(f"Map: {a.output}")
