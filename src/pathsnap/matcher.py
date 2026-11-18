"""Candidate generation, map-matching features, model comparison and Viterbi decoding."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json, time
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
from .data import Trip, read_trip

@dataclass
class CandidateBatch:
    ids: np.ndarray       # n_observations x k arc ids
    features: np.ndarray  # n_observations x k x n_features
    truth: np.ndarray     # nearest reference-route arc for each observation
    candidate_miss: np.ndarray

FEATURE_NAMES = ["distance_m", "heading_difference_rad", "along_segment_fraction", "segment_length_m", "gps_speed_mps"]


def _xy(lonlat: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    if len(lonlat)==0: return np.empty((0,2),dtype=float)
    return np.column_stack(((lonlat[:,0]-lon0)*111320*np.cos(np.deg2rad(lat0)), (lonlat[:,1]-lat0)*110540))


def _geometry(trip: Trip):
    if not len(trip.track): raise ValueError(f"Trip {trip.trip_id} has no GPS data")
    lon0=float(np.median(trip.track[:,0])); lat0=float(np.median(trip.track[:,1]))
    xy_nodes=_xy(trip.nodes,lon0,lat0)
    valid=(trip.arcs[:,0]>=0)&(trip.arcs[:,1]>=0)&(trip.arcs[:,0]<len(xy_nodes))&(trip.arcs[:,1]<len(xy_nodes))
    arc_ids=np.flatnonzero(valid)
    starts=xy_nodes[trip.arcs[valid,0]]; ends=xy_nodes[trip.arcs[valid,1]]
    vec=ends-starts; lengths=np.linalg.norm(vec,axis=1)
    keep=lengths>0
    return lon0,lat0,arc_ids[keep],starts[keep],ends[keep],vec[keep],lengths[keep]


def _point_segment_distance(points, starts, vectors, lengths):
    denom=np.maximum(lengths*lengths,1e-9)
    t=np.clip(np.einsum("...d,...d->...", points-starts, vectors)/denom,0,1)
    proj=starts+t[...,None]*vectors
    return np.linalg.norm(points-proj,axis=-1),t


def _reference_labels(trip: Trip, arc_ids, starts, vectors, lengths, gps_xy):
    valid=(trip.route>=0)&(trip.route<len(trip.arcs))
    route_ids=trip.route[valid]
    if not len(route_ids): return np.full(len(gps_xy),-1,dtype=np.int64)
    # Map route arc IDs to valid, nonzero-length geometry rows.
    pos={int(a):i for i,a in enumerate(arc_ids)}
    route_ids=np.array([int(a) for a in route_ids if int(a) in pos],dtype=np.int64)
    rows=np.array([pos[int(a)] for a in route_ids],dtype=np.int64)
    if not len(rows): return np.full(len(gps_xy),-1,dtype=np.int64)
    rs,re,rv,rl=starts[rows], starts[rows]+vectors[rows], vectors[rows], lengths[rows]
    out=np.empty(len(gps_xy),dtype=np.int64)
    # Chunked calculation bounds memory on long trips.
    for lo in range(0,len(gps_xy),128):
        pts=gps_xy[lo:lo+128,None,:]
        dist,_=_point_segment_distance(pts,rs[None,:,:],rv[None,:,:],rl[None,:])
        best=np.argmin(dist,axis=1)
        out[lo:lo+len(best)]=route_ids[np.minimum(best,len(route_ids)-1)]
    return out


def generate_candidates(trip: Trip, k: int=16) -> CandidateBatch:
    lon0,lat0,arc_ids,starts,ends,vectors,lengths=_geometry(trip)
    if not len(arc_ids): raise ValueError(f"Trip {trip.trip_id} has no valid road segments")
    gps=_xy(trip.track[:,:2],lon0,lat0)
    mids=(starts+ends)/2
    tree=cKDTree(mids)
    k=min(k,len(arc_ids)); _,ix=tree.query(gps,k=k,workers=-1)
    if k==1: ix=ix[:,None]
    cand_ids=arc_ids[ix]
    cstarts=starts[ix]; cvec=vectors[ix]; clength=lengths[ix]
    pts=np.broadcast_to(gps[:,None,:],cstarts.shape)
    dist,along=_point_segment_distance(pts,cstarts,cvec,clength)
    gps_delta=np.vstack((np.zeros((1,2)),np.diff(gps,axis=0)))
    dt=np.diff(trip.track[:,2],prepend=trip.track[0,2]); dt[dt<=0]=1.0
    gps_speed=np.linalg.norm(gps_delta,axis=1)/dt
    gps_heading=np.arctan2(gps_delta[:,1],gps_delta[:,0])
    road_heading=np.arctan2(cvec[:,:,1],cvec[:,:,0])
    diff=np.abs((road_heading-gps_heading[:,None]+np.pi)%(2*np.pi)-np.pi)
    feats=np.stack([dist,diff,along,clength,np.broadcast_to(gps_speed[:,None],dist.shape)],axis=-1).astype(np.float32)
    truth=_reference_labels(trip,arc_ids,starts,vectors,lengths,gps)
    miss=~np.any(cand_ids==truth[:,None],axis=1)
    return CandidateBatch(cand_ids,feats,truth,miss)


def _flatten_batches(batches):
    X=[]; y=[]; groups=[]
    for trip_id,b in batches:
        ok=(b.truth>=0)&(~b.candidate_miss)
        xi=b.features[ok].reshape(-1,len(FEATURE_NAMES)); ids=b.ids[ok].reshape(-1); truths=np.repeat(b.truth[ok],b.ids.shape[1])
        X.append(xi); y.append((ids==truths).astype(np.int8)); groups.extend([trip_id]*len(xi))
    return np.concatenate(X),np.concatenate(y),np.asarray(groups)


def _estimators(seed=42):
    return {
        "Logistic Regression": make_pipeline(StandardScaler(),LogisticRegression(max_iter=250,class_weight="balanced",random_state=seed)),
        "KNN": make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=9,weights="distance",n_jobs=-1)),
        "Decision Tree": DecisionTreeClassifier(max_depth=14,min_samples_leaf=4,class_weight="balanced",random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=80,max_depth=18,min_samples_leaf=2,class_weight="balanced_subsample",n_jobs=-1,random_state=seed),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(max_iter=100,learning_rate=.08,max_leaf_nodes=15,l2_regularization=1.0,random_state=seed),
    }


def _evaluate(model, batches):
    top_correct=total=0; cand_miss=0; candidate_y=[]; candidate_p=[]; t0=time.perf_counter()
    for _,b in batches:
        n,k=b.ids.shape; X=b.features.reshape(-1,b.features.shape[-1])
        if hasattr(model,"predict_proba"): p=model.predict_proba(X)[:,1]
        else: p=model.decision_function(X); p=1/(1+np.exp(-np.clip(p,-30,30)))
        p=p.reshape(n,k)
        valid=b.truth>=0
        choices=np.argmax(p,axis=1); pred=b.ids[np.arange(n),choices]
        top_correct+=int(np.sum((pred==b.truth)&valid)); total+=int(valid.sum()); cand_miss+=int(np.sum(b.candidate_miss&valid))
        candidate_y.append((b.ids[valid].reshape(-1)==np.repeat(b.truth[valid],k)).astype(np.int8))
        candidate_p.append(p[valid].reshape(-1))
    elapsed=time.perf_counter()-t0
    yy=np.concatenate(candidate_y); pp=np.concatenate(candidate_p)
    hard=(pp>=.5).astype(np.int8)
    return {"top1_accuracy":top_correct/max(total,1),"candidate_miss_rate":cand_miss/max(total,1),"candidate_macro_f1":f1_score(yy,hard,average="macro",zero_division=0),"candidate_log_loss":log_loss(yy,np.clip(pp,1e-6,1-1e-6),labels=[0,1]),"inference_seconds":elapsed,"observations":total}


def train_compare(data_dir: Path, out_dir: Path, k=12, seed=42, max_candidates=200_000):
    train_root=data_dir/"train"; test_root=data_dir/"student_test"
    train_trips=[read_trip(p,"train") for p in sorted(train_root.iterdir()) if p.is_dir() and (p/f"{p.name}.track").exists()]
    test_trips=[read_trip(p,"student_test") for p in sorted(test_root.iterdir()) if p.is_dir() and (p/f"{p.name}.track").exists()]
    train_batches=[(t.trip_id,generate_candidates(t,k)) for t in train_trips]
    test_batches=[(t.trip_id,generate_candidates(t,k)) for t in test_trips]
    # Reserve whole training trips for an initial phase-1 validation split.
    splitter=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed)
    batch_groups=np.asarray([tid for tid,_ in train_batches])
    fit_idx,val_idx=next(splitter.split(np.arange(len(train_batches)),groups=batch_groups))
    fit_batches=[train_batches[i] for i in fit_idx]; val_batches=[train_batches[i] for i in val_idx]
    X,y,groups=_flatten_batches(fit_batches)
    rng=np.random.default_rng(seed)
    pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0)
    neg_keep=rng.choice(neg,size=min(len(neg),max(4*len(pos),1)),replace=False)
    take=np.concatenate((pos,neg_keep))
    if len(take)>max_candidates: take=rng.choice(take,size=max_candidates,replace=False)
    X,y=X[take],y[take]
    rows=[]; models={}; fitted_for_validation={}
    for name,model in _estimators(seed).items():
        sample_limit=5000 if name=="KNN" else len(y)
        fit_rows=np.arange(len(y))
        if len(fit_rows)>sample_limit:
            # Balanced subsampling keeps KNN inference practical and reproducible.
            positives=fit_rows[y==1]; negatives=fit_rows[y==0]
            take_pos=rng.choice(positives,size=min(len(positives),sample_limit//2),replace=False)
            take_neg=rng.choice(negatives,size=min(len(negatives),sample_limit-len(take_pos)),replace=False)
            fit_rows=np.concatenate((take_pos,take_neg))
        start=time.perf_counter(); model.fit(X[fit_rows],y[fit_rows]); train_s=time.perf_counter()-start
        val_metrics=_evaluate(model,val_batches)
        test_metrics=_evaluate(model,test_batches)
        rows.append({"model":name,"training_seconds":train_s,
                     **{f"validation_{key}":value for key,value in val_metrics.items()},
                     **{f"student_test_{key}":value for key,value in test_metrics.items()}})
        fitted_for_validation[name]=model
    # Transparent no-learning baseline: nearest segment among generated candidates.
    def geometric_eval(batches):
        correct=seen=miss=0
        for _,batch in batches:
            ix=np.argmin(batch.features[:,:,0],axis=1); pred=batch.ids[np.arange(len(ix)),ix]; valid=batch.truth>=0
            correct+=int(np.sum((pred==batch.truth)&valid)); seen+=int(valid.sum()); miss+=int(np.sum(batch.candidate_miss&valid))
        return {"top1_accuracy":correct/max(seen,1),"candidate_miss_rate":miss/max(seen,1),"observations":seen}
    for split,batches in (("validation",val_batches),("student_test",test_batches)):
        metrics=geometric_eval(batches)
        row={"model":"Nearest road segment"}
        row.update({f"{split}_{key}":value for key,value in metrics.items()})
        if split=="validation": rows.insert(0,row)
        else: rows[0].update({f"student_test_{key}":value for key,value in metrics.items()})
    # Select by trip-held-out top-1 accuracy, then refit that choice on all 80 training trips.
    learned=[r for r in rows if r["model"]!="Nearest road segment"]
    selected=max(learned,key=lambda r:r["validation_top1_accuracy"])["model"]
    all_X,all_y,_=_flatten_batches(train_batches)
    pos=np.flatnonzero(all_y==1); neg=np.flatnonzero(all_y==0)
    neg_keep=rng.choice(neg,size=min(len(neg),max(4*len(pos),1)),replace=False)
    take=np.concatenate((pos,neg_keep))
    if len(take)>max_candidates: take=rng.choice(take,size=max_candidates,replace=False)
    final_model=_estimators(seed)[selected]
    fit_rows=np.arange(len(take))
    if selected=="KNN" and len(fit_rows)>5000: fit_rows=rng.choice(fit_rows,size=5000,replace=False)
    final_model.fit(all_X[take][fit_rows],all_y[take][fit_rows])
    out_dir.mkdir(parents=True,exist_ok=True)
    import csv
    with (out_dir/"model_comparison.csv").open("w",newline="",encoding="utf-8") as f:
        fields=list(dict.fromkeys(k for row in rows for k in row)); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    joblib.dump({"model":final_model,"model_name":selected,"feature_names":FEATURE_NAMES,"candidate_k":k,"seed":seed},out_dir/"phase1_model.joblib")
    summary={"candidate_k":k,"feature_names":FEATURE_NAMES,"training_trips":len(train_trips),"student_test_trips":len(test_trips),"training_candidate_rows":int(len(y)),"positive_training_candidates":int(y.sum()),"validation_trip_count":len(val_batches),"selected_model_by_validation_top1":selected,"results":rows}
    (out_dir/"training_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return rows,selected,final_model

def predict_batch(model, batch: CandidateBatch):
    n,k=batch.ids.shape
    p=model.predict_proba(batch.features.reshape(-1,batch.features.shape[-1]))[:,1].reshape(n,k)
    return batch.ids[np.arange(n),np.argmax(p,axis=1)],p


def viterbi_decode(candidate_ids: np.ndarray, emissions: np.ndarray, arcs: np.ndarray, gps_gaps: np.ndarray|None=None, invalid_penalty: float=8.0):
    """Decode a plausible candidate sequence using adjacency as a transition prior."""
    n,k=candidate_ids.shape
    if n==0:return np.empty(0,dtype=np.int64)
    eps=1e-9; score=np.log(np.clip(emissions,eps,1)); back=np.zeros((n,k),dtype=np.int32)
    for t in range(1,n):
        prev=candidate_ids[t-1]; cur=candidate_ids[t]
        prev_to=arcs[np.clip(prev,0,len(arcs)-1),1]; cur_from=arcs[np.clip(cur,0,len(arcs)-1),0]
        connected=(prev_to[:,None]==cur_from[None,:])
        same=prev[:,None]==cur[None,:]
        transition=np.where(connected,0.0,np.where(same,-1.5,-invalid_penalty))
        if gps_gaps is not None and gps_gaps[t]>5: transition=np.where(connected,0.0,np.where(same,-.5,-2.0))
        values=score[t-1,:,None]+transition
        back[t]=np.argmax(values,axis=0)
        score[t]=np.max(values,axis=0)+np.log(np.clip(emissions[t],eps,1))
    path=np.empty(n,dtype=np.int32); path[-1]=int(np.argmax(score[-1]))
    for t in range(n-2,-1,-1):path[t]=back[t+1,path[t+1]]
    return candidate_ids[np.arange(n),path]
