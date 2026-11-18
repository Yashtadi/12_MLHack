"""Readers and preprocessing for the headerless PathSnap trip files."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class Trip:
    trip_id: str
    partition: str
    track: np.ndarray  # cleaned longitude, latitude, timestamp_s
    nodes: np.ndarray  # longitude, latitude
    arcs: np.ndarray   # from_node_id, to_node_id
    route: np.ndarray  # chronological arc_id sequence
    raw_track_rows: int = 0


def clean_track(track: np.ndarray) -> np.ndarray:
    """Drop non-finite/out-of-range coordinates and non-increasing timestamps, preserving order."""
    if len(track)==0:return np.empty((0,3),dtype=float)
    valid=np.isfinite(track).all(axis=1)
    valid &= (np.abs(track[:,0])<=180)&(np.abs(track[:,1])<=90)
    cleaned=track[valid]
    if len(cleaned)<2:return cleaned
    # Do not reorder trajectories: discard duplicate/backward time rows as data errors.
    keep=np.ones(len(cleaned),dtype=bool)
    last=cleaned[0,2]
    for i in range(1,len(cleaned)):
        if cleaned[i,2]<=last:keep[i]=False
        else:last=cleaned[i,2]
    return cleaned[keep]


def read_trip(path: str | Path, partition: str | None = None) -> Trip:
    path=Path(path); trip_id=path.name; part=partition or path.parent.name
    def load(ext: str, cols: int) -> np.ndarray:
        f=path/f"{trip_id}.{ext}"
        if not f.is_file():return np.empty((0,cols),dtype=float)
        arr=np.loadtxt(f,dtype=float,ndmin=2)
        if arr.shape[1]!=cols:raise ValueError(f"{f}: expected {cols} columns, got {arr.shape[1]}")
        return arr
    raw_track=load("track",3)
    track=clean_track(raw_track)
    return Trip(trip_id,part,track,load("nodes",2),load("arcs",2).astype(np.int64),load("route",1).reshape(-1).astype(np.int64),len(raw_track))
