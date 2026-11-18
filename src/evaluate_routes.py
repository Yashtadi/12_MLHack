from pathlib import Path
import argparse,csv
import numpy as np,joblib
from pathsnap.data import read_trip
from pathsnap.matcher import generate_candidates,predict_batch,viterbi_decode

def continuity(ids,arcs):
    if len(ids)<2:return float('nan')
    good=(ids[:-1]>=0)&(ids[1:]>=0)
    if not np.any(good):return 0.0
    return float(np.mean(arcs[ids[:-1][good],1]==arcs[ids[1:][good],0]))

def main(data,model_path,out):
    bundle=joblib.load(model_path); rows=[]
    for p in sorted((data/'student_test').iterdir()):
        if not p.is_dir() or not (p/f'{p.name}.track').exists():continue
        trip=read_trip(p,'student_test'); b=generate_candidates(trip,bundle['candidate_k']); classifier,probs=predict_batch(bundle['model'],b)
        gaps=np.diff(trip.track[:,2],prepend=trip.track[0,2]); decoded=viterbi_decode(b.ids,probs,trip.arcs,gaps)
        valid=b.truth>=0
        rows.append({'trip_id':trip.trip_id,'observations':int(valid.sum()),'candidate_miss_rate':float(b.candidate_miss[valid].mean()),
            'classifier_top1_accuracy':float(np.mean(classifier[valid]==b.truth[valid])),'viterbi_top1_accuracy':float(np.mean(decoded[valid]==b.truth[valid])),
            'classifier_direct_arc_continuity':continuity(classifier,trip.arcs),'viterbi_direct_arc_continuity':continuity(decoded,trip.arcs)})
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    n=sum(r['observations'] for r in rows)
    summary={k:float(np.average([r[k] for r in rows],weights=[r['observations'] for r in rows])) for k in ('candidate_miss_rate','classifier_top1_accuracy','viterbi_top1_accuracy')}
    summary.update({'student_test_trips':len(rows),'observations':n,
      'classifier_direct_arc_continuity_trip_mean':float(np.mean([r['classifier_direct_arc_continuity'] for r in rows])),
      'viterbi_direct_arc_continuity_trip_mean':float(np.mean([r['viterbi_direct_arc_continuity'] for r in rows]))})
    out.with_suffix('.summary.json').write_text(__import__('json').dumps(summary,indent=2),encoding='utf-8')
    print(summary)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,default=Path('data/Student folder'));ap.add_argument('--model',type=Path,default=Path('outputs/phase1_model.joblib'));ap.add_argument('--out',type=Path,default=Path('outputs/route_decoder_comparison.csv'));a=ap.parse_args();main(a.data_dir,a.model,a.out)
