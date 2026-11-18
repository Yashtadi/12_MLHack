from pathlib import Path
import csv, json, sys
import matplotlib.pyplot as plt

def main(root=Path('.')):
    root=Path(root); outputs=root/'outputs'; reports=root/'reports'; reports.mkdir(exist_ok=True)
    with (outputs/'model_comparison.csv').open(newline='') as f: rows=list(csv.DictReader(f))
    learned=[r for r in rows if r['model']!='Nearest road segment']
    best=max(learned,key=lambda r:float(r['validation_top1_accuracy']))
    lines=['# Phase 1 working evidence','', '## Preprocessing performed','',
      '1. Read the four headerless per-trip files using fixed schemas: GPS (longitude, latitude, timestamp), nodes (longitude, latitude), arcs (from node, to node), and route arc-row IDs.',
      '2. Drop GPS rows with non-finite values or coordinates outside valid longitude/latitude bounds. Preserve observation order and drop duplicate/backward timestamps rather than sorting a trajectory.',
      '3. Keep GPS gaps in the sequence; compute speed using the actual elapsed timestamp so a gap is not mistaken for a one-second step.',
      '4. Project each trip to a local metre-scale coordinate frame before distance and segment calculations. Exclude arcs with invalid node references and zero geometric length from candidate geometry.',
      '5. Generate 12 nearby road-segment candidates per observation from segment midpoints using a spatial index, then calculate exact point-to-segment features.',
      '6. Build training labels by matching each GPS observation to the closest road segment in that trip’s supplied reference route. Candidate-miss observations are measured and omitted from binary candidate-ranker fitting because the correct segment is absent from their candidate list.',
      '', 'The supplied dataset audit found 0 invalid-coordinate rows and 0 non-increasing timestamps in the inspected partitions. So the cleaning rules ran, but did not remove rows from these files. The 1-second sampling pattern is nominal; gaps over 5 seconds are retained and reported.','',
      '## Feature choices and reasons','', '| Feature | What it describes | Why use it for road matching |','|---|---|---|',
      '| Distance to segment (m) | Perpendicular distance from GPS point to candidate road segment | Noisy GPS is usually closest to the traveled road; metre units make distance interpretable. |',
      '| Heading difference (radians) | Difference between the vehicle’s recent direction and candidate segment direction | Helps distinguish nearby parallel or oppositely directed roads. |',
      '| Along-segment fraction | Projected position between the segment endpoints, clipped to 0–1 | Indicates whether the point lies near the candidate segment’s span versus near its extension. |',
      '| Segment length (m) | Length of the candidate arc | Provides local geometry context for interpreting distance and endpoint position. |',
      '| GPS speed (m/s) | Distance moved since the previous observation divided by elapsed seconds | Adds movement context and respects irregular time gaps. |',
      '', 'This is a compact, task-motivated phase-1 feature set. A formal feature-ablation study has not yet been run, so the team should not claim these are globally optimal. Logistic Regression and KNN use standardized features; tree models use the metre/radian feature values directly.','',
      '## Five-model comparison','', 'Models were fitted on candidate rows from 64 training trips; 16 whole training trips were held out for phase-1 validation. All five learned models used the same candidate features and labels. Student-test results are shown as development-stage comparisons only. The candidate-miss rate is shared by models because candidate generation is shared.','',
      '| Model | Validation Top-1 | Validation candidate Macro F1 | Student-test Top-1 | Student-test candidate Macro F1 | Student-test log loss | Inference seconds (student test) |','|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        def pct(v):return f'{100*float(v):.2f}%' if v not in ('',None) else '—'
        def num(v):return f'{float(v):.4f}' if v not in ('',None) else '—'
        lines.append(f"| {r['model']} | {pct(r.get('validation_top1_accuracy',''))} | {num(r.get('validation_candidate_macro_f1',''))} | {pct(r.get('student_test_top1_accuracy',''))} | {num(r.get('student_test_candidate_macro_f1',''))} | {num(r.get('student_test_candidate_log_loss',''))} | {num(r.get('student_test_inference_seconds',''))} |")
    lines += ['', '### Recorded initial configurations','', '| Model | Phase-1 configuration |','|---|---|', '| Logistic Regression | StandardScaler; max_iter=250; class_weight=balanced |', '| KNN | StandardScaler; k=9; distance-weighted neighbors; 5,000 balanced candidate rows for practical inference |', '| Decision Tree | max_depth=14; min_samples_leaf=4; class_weight=balanced |', '| Random Forest | 80 trees; max_depth=18; min_samples_leaf=2; balanced_subsample |', '| Hist Gradient Boosting | 100 boosting iterations; learning_rate=0.08; max_leaf_nodes=15; L2=1.0 |', '', '### Initial model choice','',
      f"**Hist Gradient Boosting** is the provisional phase-1 selection because it had the highest whole-trip validation Top-1 accuracy ({100*float(best['validation_top1_accuracy']):.2f}%), narrowly above Random Forest ({100*float(next(r for r in learned if r['model']=='Random Forest')['validation_top1_accuracy']):.2f}%). It also had lower measured student-test inference time than Random Forest ({float(best['student_test_inference_seconds']):.3f}s vs {float(next(r for r in learned if r['model']=='Random Forest')['student_test_inference_seconds']):.3f}s) and the highest student-test Top-1 in this initial comparison ({100*float(best['student_test_top1_accuracy']):.2f}%).",
      'The validation margin over Random Forest is small (about 0.36 percentage points), so this is an initial choice, not a final claim of superiority. A later phase should use group-aware cross-validation, hyperparameter search, probability calibration, and a candidate-generation ablation before finalizing the model.',
      '', 'Candidate Macro F1 and log loss here are computed at the binary candidate-row level (correct candidate vs other candidate), not as a globally shared road-ID classification score. Arc IDs are local to each trip.','']
    route_path=outputs/'route_decoder_comparison.summary.json'
    if route_path.exists():
        r=json.loads(route_path.read_text())
        lines += ['## Initial HMM/Viterbi comparison','',f"On all {r['student_test_trips']} student-test trips ({r['observations']:,} observations), classifier Top-1 was {100*r['classifier_top1_accuracy']:.2f}% and the initial HMM/Viterbi output was {100*r['viterbi_top1_accuracy']:.2f}%. The strict direct-arc continuity rate changed from {100*r['classifier_direct_arc_continuity_trip_mean']:.2f}% to {100*r['viterbi_direct_arc_continuity_trip_mean']:.2f}% (trip-average).",'', 'This is a visible working decoder, but its first transition rule trades pointwise accuracy for direct adjacency. Treat it as an initial prototype and a clear next improvement: allow plausible multi-arc paths between GPS samples, account for elapsed time and speed, and tune decoder parameters on permitted development data. Do not describe the current decoder as improving accuracy.','']
    lines += ['## Reproducible demonstration','', 'The local web app lets a reviewer select a student-test trip, runs candidate generation → selected classifier → Viterbi decoding, and displays the predicted route, metrics, and an interactive Folium map with GPS points, reference route, candidate road context, and predicted route. The app and API run on one laptop.','', '## Phase-1 limits','', '- Evaluator-test data was not included in the supplied ZIP and was not used.', '- Initial validation is a single trip-wise holdout rather than the full group-aware cross-validation requested for the complete submission.', '- The route label for each observation is approximated by nearest geometry on the supplied route sequence; candidate misses remain a measurable ceiling.', '- The Viterbi transition model is an initial implementation; the first all-test comparison shows an accuracy/continuity tradeoff that needs improvement.', '- Feature ablations, Bayesian optimization, full final report, and the final evaluator-facing trip flow remain later work.','']
    (reports/'phase1_working_evidence.md').write_text('\n'.join(lines),encoding='utf-8')
    # Comparison plot uses measured output CSV values.
    names=[r['model'] for r in rows]; val=[float(r.get('validation_top1_accuracy') or 0) for r in rows]; test=[float(r.get('student_test_top1_accuracy') or 0) for r in rows]
    fig,ax=plt.subplots(figsize=(10,5));x=range(len(names));w=.38
    ax.bar([i-w/2 for i in x],val,width=w,label='Training-trip validation');ax.bar([i+w/2 for i in x],test,width=w,label='Student test (development)')
    ax.set_xticks(list(x),names,rotation=20,ha='right');ax.set_ylabel('Observation-level Top-1 accuracy');ax.set_ylim(0,1);ax.legend();ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(root/'results'/'model_comparison.png',dpi=160);plt.close(fig)
    print('Wrote',reports/'phase1_working_evidence.md','and',root/'results'/'model_comparison.png')
if __name__=='__main__':main(Path(sys.argv[1]) if len(sys.argv)>1 else Path('.'))
