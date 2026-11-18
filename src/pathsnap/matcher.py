"""Candidate generation, map-matching features, model comparison and Viterbi decoding."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
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
