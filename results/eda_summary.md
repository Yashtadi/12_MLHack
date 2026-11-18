# PathSnap Phase 1 — dataset exploration

Data source: `ML_hackathon/data/Student folder`. Training and student-test partitions were inspected; no evaluator partition was present in the supplied archive.

| Partition | Trips | GPS points | Nodes | Road arcs | Route arc entries |
|---|---:|---:|---:|---:|---:|
| train | 80 | 208,417 | 1,933,485 | 3,876,904 | 31,336 |

### train quality checks

- GPS observations per trip: median 1,529, range 361–19,313.
- Observations with invalid coordinates or nonfinite values: 0.
- Nonincreasing timestamp steps: 0; steps with gaps over 5 seconds: 1,216.
- Trips whose route entries all index an arc row: 80/80.
| student_test | 10 | 19,160 | 90,689 | 185,041 | 3,477 |

### student_test quality checks

- GPS observations per trip: median 1,420, range 262–3,958.
- Observations with invalid coordinates or nonfinite values: 0.
- Nonincreasing timestamp steps: 0; steps with gaps over 5 seconds: 44.
- Trips whose route entries all index an arc row: 10/10.

## Data interpretation and phase-1 assumptions

- `.track` gives longitude, latitude, and seconds; `.nodes` gives longitude and latitude; `.arcs` gives directed endpoint node IDs; `.route` is a chronological list of arc-row IDs.
- Each trip has its own road network, so candidate features describe geometry and movement rather than treating local arc IDs as a globally shared label vocabulary.
- Candidate generation uses a local metre-scale projection, a spatial tree over segment midpoints, then exact point-to-segment distances.
- Training labels are derived by associating each GPS point with the closest segment in that trip’s supplied reference route. This is a phase-1 proxy; route/reference limitations should be reviewed before final conclusions.
- Development model selection uses a whole-trip holdout from the 80 training trips. Student-test results are reported for development only. Evaluator-test data and labels are not present in the supplied archive.
