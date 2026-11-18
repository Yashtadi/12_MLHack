"""Dataset inventory and quality summaries for PathSnap."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
from .data import read_trip


def summarize_trip(path: Path, partition: str) -> dict:
    trip = read_trip(path, partition)
    t = trip.track
    dt = np.diff(t[:, 2]) if len(t) > 1 else np.array([])
    valid = np.isfinite(t).all(axis=1) if len(t) else np.array([], dtype=bool)
    valid_xy = valid & (np.abs(t[:, 0]) <= 180) & (np.abs(t[:, 1]) <= 90) if len(t) else valid
    route_in_range = int(np.sum((trip.route >= 0) & (trip.route < len(trip.arcs))))
    return {
        "partition": partition, "trip_id": trip.trip_id,
        "gps_rows": len(t), "raw_gps_rows": trip.raw_track_rows, "preprocessing_rows_removed": trip.raw_track_rows-len(t), "node_rows": len(trip.nodes), "arc_rows": len(trip.arcs),
        "route_rows": len(trip.route), "route_arc_ids_in_range": route_in_range,
        "finite_gps_rows": int(valid.sum()), "valid_coordinate_rows": int(valid_xy.sum()),
        "nonincreasing_timestamp_steps": int(np.sum(dt <= 0)),
        "one_second_steps": int(np.sum(np.isclose(dt, 1.0))),
        "time_gap_over_5s_steps": int(np.sum(dt > 5)),
        "median_step_seconds": float(np.median(dt)) if len(dt) else "",
        "longitude_min": float(np.min(t[:, 0])) if len(t) else "",
        "longitude_max": float(np.max(t[:, 0])) if len(t) else "",
        "latitude_min": float(np.min(t[:, 1])) if len(t) else "",
        "latitude_max": float(np.max(t[:, 1])) if len(t) else "",
    }


def run(data_dir: Path, out: Path) -> list[dict]:
    rows=[]
    for partition in ("train", "student_test"):
        root=data_dir / partition
        if not root.exists():
            continue
        for p in sorted(root.iterdir()):
            if p.is_dir() and (p / f"{p.name}.track").exists():
                rows.append(summarize_trip(p, partition))
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out.open("w", newline="", encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("data_dir",type=Path); ap.add_argument("--out",type=Path,default=Path("results/dataset_inventory.csv")); a=ap.parse_args()
    rows=run(a.data_dir,a.out)
    print(f"Wrote {len(rows)} trip summaries to {a.out}")
    for part in ("train","student_test"):
        subset=[r for r in rows if r["partition"]==part]
        print(f"{part}: trips={len(subset)}, GPS observations={sum(r['gps_rows'] for r in subset):,}")
