from pathlib import Path
import csv, argparse, statistics
import numpy as np
import matplotlib.pyplot as plt
from pathsnap.eda import run
from pathsnap.data import read_trip
from pathsnap.matcher import generate_candidates
from pathsnap.visualize import create_map

def main(data_dir,out_dir):
    out_dir.mkdir(parents=True,exist_ok=True)
    rows=run(data_dir,out_dir/'dataset_inventory.csv')
    lines=['# PathSnap Phase 1 — dataset exploration','',f'Data source: `{data_dir}`. Training and student-test partitions were inspected; no evaluator partition was present in the supplied archive.','', '| Partition | Trips | GPS points | Nodes | Road arcs | Route arc entries |','|---|---:|---:|---:|---:|---:|']
    for part in ('train','student_test'):
        rs=[r for r in rows if r['partition']==part]
        lines.append(f"| {part} | {len(rs)} | {sum(r['gps_rows'] for r in rs):,} | {sum(r['node_rows'] for r in rs):,} | {sum(r['arc_rows'] for r in rs):,} | {sum(r['route_rows'] for r in rs):,} |")
        if rs:
            gps=[r['gps_rows'] for r in rs];gaps=[r['time_gap_over_5s_steps'] for r in rs]
            nonmono=sum(r['nonincreasing_timestamp_steps'] for r in rs)
            invalid=sum(r['gps_rows']-r['valid_coordinate_rows'] for r in rs)
            validref=sum(r['route_rows']==r['route_arc_ids_in_range'] for r in rs)
            lines += ['',f'### {part} quality checks','',f'- GPS observations per trip: median {statistics.median(gps):,.0f}, range {min(gps):,}–{max(gps):,}.',f'- Observations with invalid coordinates or nonfinite values: {invalid:,}.',f'- Nonincreasing timestamp steps: {nonmono:,}; steps with gaps over 5 seconds: {sum(gaps):,}.',f'- Trips whose route entries all index an arc row: {validref}/{len(rs)}.']
    lines += ['', '## Data interpretation and phase-1 assumptions','', '- `.track` gives longitude, latitude, and seconds; `.nodes` gives longitude and latitude; `.arcs` gives directed endpoint node IDs; `.route` is a chronological list of arc-row IDs.', '- Each trip has its own road network, so candidate features describe geometry and movement rather than treating local arc IDs as a globally shared label vocabulary.', '- Candidate generation uses a local metre-scale projection, a spatial tree over segment midpoints, then exact point-to-segment distances.', '- Training labels are derived by associating each GPS point with the closest segment in that trip’s supplied reference route. This is a phase-1 proxy; route/reference limitations should be reviewed before final conclusions.', '- Development model selection uses a whole-trip holdout from the 80 training trips. Student-test results are reported for development only. Evaluator-test data and labels are not present in the supplied archive.','']
    (out_dir/'eda_summary.md').write_text('\n'.join(lines),encoding='utf-8')
    # Plot one route and one partition-wide GPS distribution.
    allgps=[]; sample=None
    for part in ('train','student_test'):
        root=data_dir/part
        if not root.exists():continue
        for p in sorted(root.iterdir()):
            if not p.is_dir() or not (p/f'{p.name}.track').exists():continue
            t=read_trip(p,part)
            if part=='student_test' and sample is None:sample=t
            if len(allgps)<100000:allgps.append(t.track[::max(1,len(t.track)//1000),:2])
    if allgps:
        pts=np.concatenate(allgps)
        fig,ax=plt.subplots(figsize=(8,5));ax.scatter(pts[:,0],pts[:,1],s=2,alpha=.2);ax.set(xlabel='Longitude (degrees)',ylabel='Latitude (degrees)',title='GPS coordinate distribution by trip');fig.tight_layout();fig.savefig(out_dir/'gps_distribution.png',dpi=160);plt.close(fig)
    if sample:
        batch=generate_candidates(sample,k=12); nearest=batch.ids[np.arange(len(batch.ids)),np.argmin(batch.features[:,:,0],axis=1)]
        create_map(sample,batch,nearest,out_dir.parent/'maps'/'phase1_eda_map.html')
    print('EDA outputs:',out_dir/'eda_summary.md',out_dir/'dataset_inventory.csv',out_dir/'gps_distribution.png')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,default=Path('data/Student folder'));ap.add_argument('--output-dir',type=Path,default=Path('results'));a=ap.parse_args();main(a.data_dir,a.output_dir)
