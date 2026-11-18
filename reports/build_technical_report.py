"""Build the PathSnap technical report from the committed measurements."""
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "figures"
OUT = ROOT / "reports" / "PathSnap_Phase1_Phase2_Technical_Report.pdf"
PAGE = landscape(A4)
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CoverX", parent=styles["Title"], fontSize=28, leading=34,
                          textColor=colors.HexColor("#13345b"), alignment=TA_CENTER))
styles.add(ParagraphStyle(name="SubX", parent=styles["Normal"], fontSize=12, leading=17,
                          alignment=TA_CENTER, textColor=colors.HexColor("#536579")))
styles.add(ParagraphStyle(name="H1X", parent=styles["Heading1"], fontSize=17, leading=21,
                          textColor=colors.HexColor("#13345b"), spaceAfter=8))
styles.add(ParagraphStyle(name="H2X", parent=styles["Heading2"], fontSize=11.5, leading=14,
                          textColor=colors.HexColor("#1769aa"), spaceBefore=4, spaceAfter=4))
styles.add(ParagraphStyle(name="BX", parent=styles["BodyText"], fontSize=8.6, leading=11.3, spaceAfter=5))
styles.add(ParagraphStyle(name="SX", parent=styles["BodyText"], fontSize=7.2, leading=9,
                          textColor=colors.HexColor("#526273"), spaceAfter=3))
styles.add(ParagraphStyle(name="CellX", parent=styles["BodyText"], fontSize=6.7, leading=8))

def p(text, style="BX"):
    return Paragraph(text, styles[style])

def fig(name, width=175*mm):
    image = Image(str(ASSETS / name))
    return Image(str(ASSETS / name), width=width, height=width * image.imageHeight / image.imageWidth)

def table(rows, widths, font=7.5):
    rows = [[v if isinstance(v, Paragraph) else Paragraph(str(v), styles["CellX"]) for v in row] for row in rows]
    t = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17385e")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f3f7fb")]),
        ("FONTSIZE", (0,0), (-1,-1), font), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 4), ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    return t

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#728096"))
    canvas.drawString(15*mm, 9*mm, "PathSnap | Combined Phase 1 + Phase 2 technical report")
    canvas.drawRightString(PAGE[0]-15*mm, 9*mm, str(doc.page))
    canvas.restoreState()

doc = SimpleDocTemplate(str(OUT), pagesize=PAGE, rightMargin=15*mm, leftMargin=15*mm,
                        topMargin=13*mm, bottomMargin=15*mm,
                        title="PathSnap Phase 1 and Phase 2 Technical Report", author="PathSnap Team")
story = []

# Cover
story += [Spacer(1, 23*mm), p("PATHSNAP", "CoverX"),
          p("Combined Phase 1 + Phase 2 | Technical Report and Working Evaluation", "SubX"),
          Spacer(1, 8*mm), p("GPS map matching from noisy observations to a connected road route", "SubX"),
          Spacer(1, 12*mm), p("What this phase delivers", "H1X"),
          p("Phase 1 built the first preprocessing, candidate-generation and five-model baseline using a whole-trip holdout. Phase 2 strengthened model selection with grouped cross-validation, tuned Hist Gradient Boosting, added route decoding, and delivered the evaluator-facing API/UI. The API keeps reference labels on Laptop 1 and returns predictions and server-computed scores to the browser."),
          Spacer(1, 5*mm), p("Phase 1 provisional winner: Hist Gradient Boosting by validation Top-1 | Phase 2 winner: Hist Gradient Boosting by grouped-CV candidate Macro F1 | Optuna TPE: 15 trials"),
          Spacer(1, 8*mm), p("This single report distinguishes the two phases and their evaluation protocols, as well as development partitions from the reserved evaluator partition. It records measured tradeoffs and limitations rather than presenting one score as proof of complete route correctness.", "SubX"), PageBreak()]

# Dataset and preprocessing
story += [p("1. Problem, data and preprocessing", "H1X"),
          p("PathSnap receives a time-ordered GPS track and a trip-specific directed road graph. For every observation, it proposes nearby road arcs, scores them, and reconstructs a plausible sequence. The supplied files are <font name='Courier'>.track</font> (GPS coordinates/time), <font name='Courier'>.nodes</font> (road endpoints), <font name='Courier'>.arcs</font> (directed connections), and <font name='Courier'>.route</font> (reference route)."),
          p("Partition use: 80 training trips are used for model fitting and validation; 10 student-test trips are used for development comparison; 10 separate evaluator trips are reserved for the Phase 2 server-side final check. The evaluator archive was not available during Phase 1. In Phase 2, evaluator labels are not given to the prediction model or browser."),
          p("Preprocessing validates the file structure, numeric values, coordinate ranges, timestamp order, time gaps, and arc endpoint references. Track rows are kept in time order. Coordinates are projected into local metres for geometry calculations; longitude and latitude are retained for maps. In the recorded inventory, no GPS rows were removed, no invalid/non-finite coordinates were found, and no non-increasing timestamp steps were found."),
          fig("eda_partition_sizes.png", 118*mm),
          p("Figure 1. Recorded partition sizes. The data inventory is in <font name='Courier'>outputs/dataset_inventory.csv</font>; the exploration and preprocessing steps are shown in notebooks 01 and 02.", "SX"),
          p("Time sampling is mostly one second per point. Longer gaps are retained and recorded rather than interpolated away. This matters because route transitions across a long observation gap are less certain than transitions between adjacent one-second samples."), PageBreak()]

# candidates and features
story += [p("2. Candidate generation and feature choices", "H1X"),
          p("The full road graph may contain many arcs, so the pipeline first keeps the 12 nearest road candidates for each GPS observation, ranked by midpoint proximity. A classifier then scores those candidates. If the reference arc is absent from the shortlist, no later classifier can select it; this is measured as candidate miss rate."),
          table([["Feature", "Plain meaning", "Why useful", "Phase 1 permutation importance"],
                 ["distance_m", "GPS-to-segment distance", "Nearby candidates are generally more plausible", "0.1944 ± 0.0042 F1 drop"],
                 ["heading_difference_rad", "Difference between movement direction and road direction", "Rejects nearby roads pointing the wrong way", "0.1166 ± 0.0043"],
                 ["along_segment_fraction", "Projection position between segment endpoints (0–1)", "Locates the point along the candidate arc", "0.1977 ± 0.0060"],
                 ["segment_length_m", "Length of the road arc", "Provides a geometric cue about candidate shape", "0.0054 ± 0.0021"],
                 ["gps_speed_mps", "Speed since the preceding observation", "Adds motion context", "0.0070 ± 0.0014"]],
                [37*mm, 61*mm, 91*mm, 53*mm]),
          Spacer(1, 3*mm), fig("feature_importance.png", 130*mm),
          p("Figure 2. Permutation importance from Phase 1. It measures the change in Macro F1 when a feature is shuffled; it is a diagnostic, not proof of causality. All five features were retained for a consistent model comparison.", "SX"),
          p("Training labels are approximated by assigning each GPS point to the nearest geometric segment among its supplied reference route. Intersections and loops can make that label ambiguous. Candidate Macro F1 therefore measures candidate classification against these approximate labels, not exact route-level correctness."), PageBreak()]

# Phase 1 initial model training and selection
story += [p("3. Phase 1: first model training and selection", "H1X"),
          p("Round 1 used a single whole-trip holdout from the 80 training trips: 64 trips for fitting and 16 for validation. GroupShuffleSplit used seed 42, preventing points from the same trip from appearing in both sets. All five models used the same candidate features and labels; all positive candidates were retained and up to four negative candidates per positive were sampled, with a 200,000-row cap. KNN used an additional balanced sample of up to 5,000 fit rows."),
          p("The Round 1 selection rule was highest validation observation Top-1. Hist Gradient Boosting scored 71.45%, narrowly ahead of Random Forest at 71.09% (0.36 percentage points). HGB also had the highest validation candidate Macro F1 and lowest candidate log loss in this run. After selection, it was refit on all 80 training trips and saved as <font name='Courier'>outputs/phase1_model.joblib</font>. Student-test metrics were development comparisons, not the selection rule."),
          table([["Learned model", "Validation Top-1", "Validation candidate\nMacro F1", "Validation log\nloss", "Student Top-1", "Student candidate\nMacro F1", "Student log\nloss", "Student inference\nsec / 19,160 obs"],
                 ["Logistic Regression", "55.22%", "0.6301", "0.4178", "56.50%", "0.6022", "0.4722", "0.0048"],
                 ["KNN", "60.79%", "0.6965", "0.5694", "65.69%", "0.7251", "0.5413", "0.4895"],
                 ["Decision Tree", "68.87%", "0.7989", "0.2398", "72.94%", "0.8024", "0.2230", "0.0074"],
                 ["Random Forest", "71.09%", "0.8323", "0.1451", "75.14%", "0.8414", "0.1385", "0.1887"],
                 ["Hist Gradient Boosting", "71.45%", "0.8478", "0.1134", "75.45%", "0.8609", "0.1065", "0.1176"]],
                [37*mm, 27*mm, 32*mm, 25*mm, 24*mm, 31*mm, 24*mm, 35*mm], 6.5),
          Spacer(1, 3*mm), fig("phase1_model_comparison.png", 142*mm),
          p("Figure 3. Round 1 validation and student-development metrics. The non-learning nearest-road baseline reached 55.37% validation Top-1 and 53.48% student Top-1. Validation candidate miss rate was 7.57% and student-test miss rate 6.35%; candidate misses come from the shared candidate generator, not from a particular classifier.", "SX"),
          p("Phase 1 settings: Logistic Regression used StandardScaler, max_iter=250, balanced classes and seed 42; KNN used StandardScaler, 9 distance-weighted neighbors and a balanced fit sample of up to 5,000 rows; Decision Tree used depth 14, leaf size 4 and balanced classes; Random Forest used 80 trees, depth 18, leaf size 2 and balanced subsampling; HGB used 100 iterations, learning rate 0.08, 15 leaf nodes and L2=1.0. Seed was 42 for randomized estimators. Section 5 records the Phase 2 comparison settings; the main listed change is Logistic Regression max_iter=300 in that phase."),
          p("Phase 1 initial route decoding (student-test): classifier Top-1 75.45% versus Viterbi 57.64%; trip-mean direct continuity 25.41% versus 73.86%. The evaluator-only archive was not available in Phase 1, so no Phase 1 evaluator score is claimed. Phase 1 used a single holdout; Phase 2 later used five-fold GroupKFold, so the two phase scores are not directly interchangeable."), PageBreak()]

# Phase 2 five model comparison table
story += [p("4. Phase 2: five-model grouped-CV comparison", "H1X"),
          p("Training rows are candidate pairs. The positive candidate is the nearest route segment; other candidates are negatives. Each fold uses positive rows and up to four sampled negatives per positive, capped at 200,000 rows; KNN is additionally fit on a balanced 5,000-row sample. Sampling seed is 42. Five-fold GroupKFold splits by trip, so observations from one trip do not leak across train/validation."),
          p("The selected objective was mean CV candidate Macro F1. Top-1 is a separate observation-level metric: choose the highest-scored candidate for each GPS point. Direct continuity checks whether consecutive chosen arcs connect directly."),
          table([["Model", "CV Macro F1\nmean ± SD", "CV Top-1", "CV direct\ncontinuity", "Student Top-1", "Student candidate\nMacro F1", "Student log\nloss", "Student direct\ncontinuity", "Inference\nsec / 19,160 obs"],
                 ["Logistic Regression", "0.6358 ± 0.0382", "50.23%", "24.25%", "57.80%", "0.6088", "0.4637", "21.97%", "0.0046"],
                 ["KNN", "0.7202 ± 0.0188", "59.53%", "25.31%", "66.11%", "0.7233", "0.5437", "23.87%", "0.4632"],
                 ["Decision Tree", "0.7981 ± 0.0143", "65.76%", "27.12%", "73.32%", "0.7980", "0.2287", "25.21%", "0.0070"],
                 ["Random Forest", "0.8327 ± 0.0183", "68.47%", "27.44%", "75.23%", "0.8413", "0.1400", "25.52%", "0.1939"],
                 ["Hist Gradient Boosting", "0.8485 ± 0.0251", "67.87%", "27.51%", "75.44%", "0.8613", "0.1067", "25.39%", "0.1418"]],
                [37*mm, 30*mm, 22*mm, 26*mm, 23*mm, 29*mm, 22*mm, 28*mm, 27*mm], 6.7),
          Spacer(1, 3*mm), fig("cv_metric_comparison.png", 130*mm),
          p("Figure 4. Grouped-CV candidate Macro F1 and observation Top-1. Random Forest has slightly higher CV Top-1 than HGB; HGB wins the declared Macro F1 objective. Student-test is a development comparison, not the model-selection objective.", "SX"), PageBreak()]

# per model config/rationale
story += [p("5. Individual model settings and interpretation", "H1X"),
          p("5.1 Logistic Regression", "H2X"),
          p("Suitability: simple probability baseline. Phase 2 config: StandardScaler, C=1, max_iter=300, balanced classes, seed 42. It learns a linear boundary by minimizing binary logistic (cross-entropy) loss. Strength: fast and interpretable; limitation: cannot represent complex nonlinear feature interactions. Student: Top-1 57.80%, candidate Macro F1 0.6088, log loss 0.4637; grouped-CV Macro F1 0.6358 ± 0.0382. No matched training score was recorded, so a train-to-validation overfit gap cannot be quantified."),
          p("5.2 K-Nearest Neighbors", "H2X"),
          p("Suitability: local-neighborhood baseline. Config: StandardScaler, 9 neighbors, distance weighting, Euclidean distance (p=2), balanced fit sample capped at 5,000 rows. It uses nearby examples rather than optimizing a parametric training loss. Strength: flexible local decision boundary; limitation: prediction can be slow and probabilities less well calibrated. Student: Top-1 66.11%, Macro F1 0.7233, log loss 0.5437; CV Macro F1 0.7202 ± 0.0188; inference 0.4632 seconds. No matched training score was recorded."),
          p("5.3 Decision Tree", "H2X"),
          p("Suitability: interpretable nonlinear baseline. Config: max_depth=14, min_samples_leaf=4, balanced classes, seed 42. It chooses feature thresholds using Gini impurity. Strength: simple rules and nonlinear splits; limitation: a single tree can be unstable and overfit. Student: Top-1 73.32%, Macro F1 0.7980, log loss 0.2287; CV Macro F1 0.7981 ± 0.0143. No matched training score was recorded, so overfit is a risk to monitor, not a measured conclusion."),
          p("5.4 Random Forest", "H2X"),
          p("Suitability: robust ensemble of trees. Config: 80 trees, max_depth=18, min_samples_leaf=2, sqrt feature sampling, balanced_subsample class weights, seed 42. It averages bootstrapped trees trained with impurity-based splits. Strength: reduces the instability of one tree; limitation: more computation and less interpretability. Student: Top-1 75.23%, Macro F1 0.8413, log loss 0.1400; CV Macro F1 0.8327 ± 0.0183. CV Top-1 68.47% was slightly above HGB; no matched training score was recorded."),
          p("5.5 Hist Gradient Boosting", "H2X"),
          p("Suitability: nonlinear tabular candidate scoring. Untuned config: 100 iterations, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=20, L2=1, seed 42. It adds trees sequentially to reduce binary log loss. Strength: captured useful feature interactions and achieved the highest CV Macro F1; limitation: less transparent and can need tuning. Student: Top-1 75.44%, Macro F1 0.8613, log loss 0.1067; CV Macro F1 0.8485 ± 0.0251. No matched training score was recorded, so do not claim a measured overfit gap. Tuned results are in Section 6."),
          Spacer(1, 2*mm), fig("student_model_comparison.png", 123*mm),
          p("Figure 5. Student-set Top-1, candidate Macro F1 and log loss; the chart uses untuned outputs for a like-for-like five-model comparison.", "SX"),
          PageBreak()]

# Bayesian tuning
story += [p("6. Bayesian hyperparameter search", "H1X"),
          p("After selecting the HGB family, Optuna TPE searched 15 configurations. Each trial used the same five grouped training folds, and the objective was mean candidate Macro F1. Student-test and evaluator labels were excluded from the tuning objective."),
          table([["Parameter", "Search range"], ["max_iter", "60 to 240, step 20"], ["learning_rate", "0.025 to 0.20, log scale"], ["max_leaf_nodes", "7 to 31"], ["min_samples_leaf", "10 to 60"], ["l2_regularization", "0.0001 to 10, log scale"]], [55*mm, 75*mm]),
          Spacer(1, 3*mm), p("Best trial: max_iter=60; learning_rate=0.0259069; max_leaf_nodes=31; min_samples_leaf=47; l2_regularization=0.0487273."),
          fig("phase2_tuning_f1.png", 100*mm),
          table([["Measurement", "Untuned CV winner", "Bayesian tuned", "Interpretation"],
                 ["CV Macro F1 mean ± SD", "0.8485 ± 0.0251", "0.8516 ± 0.0279", "Small increase; fold SD also increased."],
                 ["Student Top-1", "75.44%", "75.40%", "Tiny decrease."],
                 ["Student candidate Macro F1", "0.8613", "0.8667", "Increase."],
                 ["Student precision / recall", "0.6547 / 0.8720", "0.6914 / 0.8348", "Precision up, recall down."],
                 ["Student log loss", "0.1067", "0.1352", "Worse probability loss."],
                 ["Student inference seconds", "0.1049", "0.0747", "Faster on this measured batch."]], [53*mm, 37*mm, 37*mm, 78*mm]),
          p("The tuned artifact was retained because it improves the declared grouped-CV objective and candidate Macro F1. It is not better on every metric; Top-1 and log loss are slightly worse on student-test. The search is modest (15 trials, one grouped-CV design), not an exhaustive or nested search.", "SX"), PageBreak()]

# HMM and evaluation
story += [p("7. Route reconstruction: Viterbi-style decoder", "H1X"),
          p("For each time step, hidden states are the 12 candidate arcs. The classifier probability provides an emission score: how well the candidate explains this GPS point. The transition score favors directed connectivity between consecutive arcs, mildly penalizes staying on the same arc, and penalizes disconnected arcs. For a time gap over five seconds, the disconnected penalty is relaxed. Dynamic programming stores best cumulative scores and backpointers, then traces back the best candidate sequence."),
          p("This is an initial HMM-style decoder, not a full shortest-path map matcher. It considers direct arc adjacency only; it does not search for intermediate road arcs between two GPS observations."),
          table([["Partition / measure", "Classifier only", "After Viterbi", "Change"],
                 ["Student, observation Top-1 (19,160 observations)", "75.40%", "56.10%", "Down 19.30 points"],
                 ["Student, trip-mean direct continuity", "25.69%", "77.80%", "Up 52.11 points"],
                 ["Evaluator, observation Top-1 (19,674 observations)", "69.49%", "54.36%", "Down 15.13 points"],
                 ["Evaluator, trip-mean direct continuity", "30.12%", "81.40%", "Up 51.28 points"],
                 ["Evaluator candidate miss rate", "4.79%", "—", "Shortlist coverage limit"]], [90*mm, 45*mm, 45*mm, 55*mm]),
          Spacer(1, 3*mm), fig("evaluator_metrics.png", 118*mm),
          p("Figure 6. Final frozen evaluator aggregate. This one-time server-side evaluation covers all 10 reserved trips. Those labels were not used to adjust model, features, thresholds, or decoder after the run.", "SX"),
          p("Interpretation: route continuity improves sharply while pointwise accuracy drops. The current decoder overvalues local connectivity and needs a shortest-path transition model to improve the tradeoff. Scores use nearest-route-geometry labels and are not exact route-level accuracy."), PageBreak()]

# Folium and app
story += [p("8. Maps, API/UI and two-laptop operation", "H1X"),
          p("The API runs on Laptop 1 and the browser UI on Laptop 2. Both join the same local network. Laptop 2 connects to <font name='Courier'>http://&lt;Laptop-1-LAN-IP&gt;:8000</font>, receives the evaluator trip IDs, and sends the selected ID to <font name='Courier'>POST /api/predict/{trip_id}</font>. Laptop 1 constructs an inference copy with an empty route array, generates candidates, predicts, decodes, and creates a Folium map. Only afterward does the server compare predictions with its reference route and return aggregate metrics."),
          p("The evaluator UI shows observed GPS, candidate-road context (toggleable), predicted route and the map layer controls. The interactive generated map is stored by the API under <font name='Courier'>maps/evaluator_&lt;trip-id&gt;.html</font>; the source is in <font name='Courier'>src/pathsnap/visualize.py</font>."),
          Table([[fig("laptop2_connected.png", 121*mm), fig("laptop2_evaluator_result_map.png", 121*mm)]],
                colWidths=[128*mm, 128*mm], style=TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),
                                                               ("LEFTPADDING",(0,0),(-1,-1),2),
                                                               ("RIGHTPADDING",(0,0),(-1,-1),2),
                                                               ("TOPPADDING",(0,0),(-1,-1),2),
                                                               ("BOTTOMPADDING",(0,0),(-1,-1),2)])),
          p("Figures 7–8. User-provided Laptop 2 evidence: connected UI listing the ten reserved trips (left); completed evaluation of trip 00000015 with result metrics, Folium map and layer controls (right). The displayed single-trip metrics are distinct from the 10-trip aggregate in Section 6.", "SX"),
          p("The interactive map is generated by <font name='Courier'>src/pathsnap/visualize.py</font> and served from <font name='Courier'>maps/evaluator_&lt;trip-id&gt;.html</font>. Candidate segments, observed GPS and predicted route are toggleable; the evaluator reference route is not included."),
          PageBreak(), p("Live API request logs and demo operation", "H1X"),
          p("The server was restarted in the foreground with access logging enabled, then the in-app browser on Laptop 1 connected through its LAN address. The following entries appeared in the running API terminal:"),
          table([["Observed API terminal entry", "Meaning"],
                 ["10.1.33.83 — GET /api/trips — 200 OK", "The browser reached the API and loaded the 10 evaluator IDs."],
                 ["10.1.33.83 — POST /api/predict/00000015 — 200 OK", "The browser submitted a real trip ID; the API returned the prediction."],
                 ["10.1.33.83 — GET /maps/evaluator_00000015.html — 200 OK", "The browser loaded the generated route map from the API host."]], [120*mm, 115*mm]),
          p("The API terminal log in the table is from a real foreground run, but its source address (10.1.33.83) matches Laptop 1's LAN address. This verifies that access logs work; it does not prove an independent Laptop 2 request. The original terminal image supplied showed a duplicate server start failing because port 8000 was already occupied; it is not success evidence. For final two-laptop proof, submit a trip from the actual Laptop 2, confirm the Laptop 1 log shows Laptop 2's different LAN address and POST 200 OK, and capture that terminal."),
          p("Laptop 1 run steps", "H2X"),
          p("From the project folder, activate the shared environment, set <font name='Courier'>PATHSNAP_EVALUATOR_DATA=data/evaluator_test</font>, then run <font name='Courier'>.venv/bin/python -m uvicorn pathsnap.server:app --app-dir src --host 0.0.0.0 --port 8000 --access-log --log-level info</font>. Leave this terminal open. Do not start another server on port 8000; a duplicate start produces the address-in-use error shown in the discarded screenshot."),
          p("Laptop 2 run steps", "H2X"),
          p("Serve the UI folder on Laptop 2 (for example, <font name='Courier'>python -m http.server 8080 --directory web</font>), open the UI in a browser, enter Laptop 1's LAN IP followed by <font name='Courier'>:8000</font>, and click Connect. Select an evaluator trip and click Evaluate trip. The Laptop 1 terminal should show <font name='Courier'>POST /api/predict/&lt;trip-id&gt; 200 OK</font>; the Laptop 2 page should show metrics and the predicted route."), PageBreak()]

# Repro and deliverables
story += [p("9. Reproduction, deliverables and limitations", "H1X"),
          p("Reproduction path in the repository:"),
          table([["Stage", "File to show", "Result"],
                 ["Explore dataset", "notebooks/01_Dataset_Exploration.ipynb", "Inventory, distributions, trajectory plots, Folium exploration."],
                 ["Validate and prepare", "notebooks/02_Data_Preprocessing.ipynb", "File integrity, timestamp checks, coordinate handling."],
                 ["Candidates and features", "notebooks/03_Candidate_Generation_Features.ipynb", "12-nearest candidates, feature calculation and labels."],
                 ["Phase 1 model training", "notebooks/04_ML_Model_Training.ipynb", "Five baseline model families and fitting."],
                 ["Initial evaluation", "notebooks/05_Initial_Evaluation.ipynb", "Phase 1 results and visual comparisons."],
                 ["Phase 2 validation/tuning", "notebooks/06_Phase2_Model_Validation_Tuning.ipynb", "GroupKFold, model comparison, Optuna trials, final student/evaluator evidence."],
                 ["Demo service", "src/pathsnap/server.py, src/pathsnap/matcher.py, web/index.html", "FastAPI inference, candidate scoring, decoder and browser UI."],
                 ["Setup", "README.md, requirements.txt", "Installation and two-laptop network instructions."]], [59*mm, 106*mm, 92*mm]),
          Spacer(1, 4*mm), p("Evidence files include outputs/phase2_cv_model_comparison.csv, phase2_student_comparison.csv, phase2_bayesian_trials.csv, phase2_tuned_comparison.csv, route_decoder_comparison.csv, and the frozen model outputs/phase2_model.joblib. The evaluator archive and raw evaluator route labels remain local and are excluded from Git."),
          p("Known limits", "H2X"),
          p("(1) Reference labels are approximated by nearest route geometry. (2) A 12-arc midpoint shortlist can miss the correct arc. (3) The tuning search used only 15 trials and one grouped-CV design. (4) The Viterbi-style decoder uses direct adjacency rather than shortest multi-arc paths. (5) Student and evaluator point accuracy both decrease after decoding, even though connectivity improves. (6) A successful API log was verified live, but the exact terminal screenshot still needs to be inserted into this PDF for the screenshot-specific report requirement."),
          p("Academic integrity note", "H2X"),
          p("Team members should review and understand the notebooks and results, explain the implementation in their own words, and disclose AI assistance according to course policy."),
          p("The report builder reads committed output tables and figure assets. Rebuild with <font name='Courier'>.venv/bin/python reports/build_technical_report.py</font> after updating those sources.", "SX")]

doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
