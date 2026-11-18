# PathSnap

PathSnap matches noisy GPS readings to directed road segments with Gradient Boosting and reconstructs connected routes with HMM/Viterbi. The current pipeline reached **94.01% pointwise accuracy** against inferred reference labels and **99.97% route continuity** across ten student-test trips. [Current results](results/refinement/comparison.json)

This repository includes model training, evaluation, and route reconstruction. The two-laptop demo and technical report are pending. Use the supplied dataset locally to reproduce the experiments and generate maps.

Open `01_trip_baseline.ipynb` to work through trip 00000001. The executed notebook saves its map to `results/trip_00000001.html`. Open the HTML in a browser with internet access for the map tiles and Leaflet libraries. Purple markers show ambiguous errors. Orange lines are the independent nearest-road predictions.

## Reproduce

The project was tested with Python 3.14. From this directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python roads.py .cache
python compare.py .cache
```

The default dataset location is `~/Downloads/Student folder`. Set `PATHSNAP_DATA` to override it. In Jupyter or VS Code, select the project environment for the notebook. The cache contains development labels and is for offline experiments only.

## Reading the results

- Five candidates per GPS reading are ranked by point-to-segment distance in a trip-local metric projection. Ties use arc ID. Heading uses a short centered time window, so results apply to reconstruction of a completed trip.
- Offline labels minimize cumulative GPS-to-segment distance over nondecreasing indices of the supplied reference route. Alignment may skip segments. Stops, repeated geometry, and contradictory road and GPS data can make these labels ambiguous. Predictor features are generated before reading the reference route.
- Three group-disjoint folds cover all 80 training trips. Each fit uses a seeded sample of up to 60,000 whole observations, keeping all five candidates. The final comparison fits from the full training partition using the same sample cap and evaluates every observation on the ten student-test trips. No evaluator data is involved.
- The four conventional models see distance, heading agreement, speed, heading availability, road length, direction, and endpoint degrees. The one-layer directed GraphSAGE model additionally aggregates the mean incoming and outgoing road features. Its no-neighbor ablation keeps the same scoring head and training schedule.
- `results/comparison.csv` gives fold and student-test results. Top-1 measures correct arc IDs across **all** readings, including candidate misses. Macro F1 averages the two candidate-level classes after selecting one road per reading. Binary log loss assesses each candidate's predicted match probability. Inference times cover candidate scoring only. The nearest-road rule has no fitted probability model.
- `results/per_trip.csv` gives trip-level accuracy and candidate misses. Overall Top-1 weights each GPS reading equally, so long trips have more influence.

## Improving the classifier

The follow-up experiment in `improve.py` keeps the original five candidate IDs, reference labels, and three trip-grouped folds. It adds the GPS projection along each road, distances to road endpoints, distance relative to the closest road, and heading and road-distance features from surrounding observations at four window sizes. These features use a completed trip.

All training observations are used. The experiment compares the original Logistic Regression features, richer Logistic Regression features, richer histogram boosting, a candidate-ranking MLP, and directed GraphSAGE. Neural models use candidate-wise cross-entropy on observations whose reference is present. All observations, including reference-absent cases, remain in the accuracy denominator.

Run from this directory after generating the initial cache:

```sh
python improve.py .cache/enhanced --prepare-from .cache
```

Selection uses mean cross-validation Macro F1. The chosen model is refitted and evaluated on all ten student-test trips. The selection record, fold metrics, student-test predictions, and fitted artifact are saved in `results`. Candidate Macro F1 uses the same definition as the first comparison. Neural softmax log loss and classifier binary-probability log loss have different probability interpretations, so model selection uses Macro F1.

To predict with the saved selected model:

```python
from improve import predict_trip
from roads import ROOT

arc_ids, scores = predict_trip(ROOT / "student_test" / "00000000")
```

This prediction entry point reads the GPS and road-network files. Reference-route files are used by the offline labeling and evaluation code.

## Bayesian hyperparameter tuning

Run the tuning study after building the enhanced feature cache:

```sh
python tune.py .cache/enhanced --trials 24
```

The study evaluates the original configuration, five random initial configurations, and eighteen Gaussian-process Bayesian proposals. A Matérn kernel models mean cross-validation Macro F1. Each Bayesian proposal maximizes expected improvement over 2,048 sampled configurations. The saved study includes every proposed setting, all three fold results, the trip assignments, the search space, random seed, and a fingerprint of the training cache. Repeating the command resumes completed trials. A partially completed trial runs again.

The search varies learning rate, training iterations, leaf count, minimum leaf samples, L2 regularisation, and histogram bins. Every fit uses all permitted training observations in that fold. Early stopping is disabled, so internal observation-level validation splits cannot mix readings from one trip. The original five-candidate evaluation includes reference-absent readings.

The highest mean CV Macro F1 determines the selected configuration before student-test features are loaded. The selected and original models are each fitted on the full training partition and evaluated on the same student-test readings. Candidate-scoring time is the median of five runs after a warmup, using two CPU threads. Feature generation and route decoding are outside that timing.

Artifacts are saved in `results/tuning`. `study.json` contains the search history, `selection.json` records the CV selection, `comparison.json` contains untuned and selected results including per-trip metrics, and `predictions.npz` retains the comparison probabilities. `selected_model.joblib` and `model_metadata.json` provide the fitted model and its feature schema. The existing `improve.predict_trip` entry point uses the selected tuning artifact when present.

The completed 24-configuration study selected a learning rate of 0.10667400467040687, 175 boosting iterations, 15 leaves, a minimum of 160 samples per leaf, L2 regularisation of 0.133403418196077, and 63 histogram bins. The original baseline plus five random initialisations were followed by 18 Bayesian proposals. The study and final comparison took about 17.3 minutes in this run.

| Metric | Untuned | Selected |
|---|---:|---:|
| Mean CV candidate Macro F1 | 0.908407 | 0.909517 |
| Mean CV Top-1 accuracy | 80.1595% | 80.3425% |
| Student-test Top-1 accuracy | 90.2349% | 90.3862% |
| Student-test candidate Macro F1 | 0.943911 | 0.944863 |
| Student-test binary log loss | 0.098372 | 0.098249 |
| Median scoring time for all student-test candidates | 0.1666 seconds | 0.1646 seconds |

The selected model correctly assigns 29 additional GPS readings overall. Five student-test trips improved, four declined, and one was unchanged. Scores measure agreement with the same inferred reference labels used in the earlier experiments. The saved selected model reproduced all ten trips' stored probabilities through `improve.predict_trip` using GPS and map files alone.

## HMM and Viterbi route reconstruction

Run the decoder experiment from this directory:

```sh
python decoder_experiment.py .cache/enhanced
python -m unittest test_decoder.py
```

The experiment creates out-of-fold scores for all 80 training trips using the tuned classifier and the same three trip-grouped folds. Each classifier fit excludes every observation from its validation trips. Decoder selection compares 12 combinations of transition distance scale (10, 30, or 100 metres) and transition strength (0.1, 0.3, 1, or 3), maximizing mean validation candidate Macro F1. Student-test results are computed after selection. Geometry and out-of-fold scores are cached in `.cache/decoder`.

The hidden state at a reading is its candidate directed road segment, with the GPS point projected onto that segment. Emission scores are the classifier's candidate match probabilities, normalized across that reading's candidates. Probabilities have a numerical floor of 1e-12. These are discriminative observation scores.

For each pair of consecutive readings, the decoder computes directed road distance between candidate projections. It traverses the remainder of the first segment, the shortest connecting road path, and the beginning of the next segment. It also handles movement along the same segment. Transition weights are proportional to `exp(-abs(road_distance - GPS_displacement) / beta_m)` and normalized over the reachable next candidates for each previous candidate. Transition log probabilities are multiplied by the chosen strength. Viterbi maximizes the sum of emission log scores and weighted transition log scores, storing backpointers to recover the full sequence. Row score offsets keep the recurrence numerically stable. The road-distance transition approach follows the map-matching formulation described by Newson and Krumm, adapted here to the learned observation scores. [Original paper](https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/)

The fixed routing policy treats gaps over 30 seconds and non-increasing timestamps as section boundaries. It permits road distance up to `55 * elapsed_seconds + 100` metres between readings. Backwards projection changes up to 20 metres on the same directed segment accommodate GPS jitter. Larger backwards changes require an actual directed connecting path. These routing limits were chosen as engineering assumptions. The validation search tunes beta and strength.

Candidates with ID -1 or invalid scores are unavailable. If a reading has no available candidates, its output is -1 and the next reading starts a new section. If every reachable chain dies, the decoder backtracks the previous section and records an `unreachable_candidates` break before restarting. Intermediate roads are inserted using the graph's original arc IDs. Separate route sections retain gaps and disconnections in the JSON and map output.

Use the prediction entry point on any supplied-format completed trip:

```python
from decoder import match_trip
from roads import ROOT

result = match_trip(ROOT / 'student_test' / '00000000')
```

The result contains one classifier arc ID and one decoded arc ID per GPS observation, chronological road segments for each connected section, longitude/latitude coordinates, and explicit break reasons. The prediction function reads GPS and map files, the classifier artifact, and the saved decoder configuration. Offline evaluation code separately reads reference-route information. `improve.predict_trip` continues to return the classifier-only prediction for comparison.

The selected decoder uses beta 10 metres and transition strength 0.3. Results on the 19,160 student-test readings are given below.

| Metric | Tuned classifier | Classifier plus HMM/Viterbi |
|---|---:|---:|
| Mean training-validation candidate Macro F1 | 0.909517 | 0.916825 |
| Mean training-validation Top-1 accuracy | 80.3425% | 81.4648% |
| Student-test Top-1 accuracy | 90.3862% | 92.6148% |
| Student-test candidate Macro F1 | 0.944863 | 0.958881 |
| Student-test route continuity | 96.7412% | 99.8642% |
| Mean trip arc-sequence similarity | 0.623625 | 0.781190 |

Route continuity is the fraction of consecutive predictions connected by a directed path within the routing distance limit. Its denominator includes decoder resets and missing-candidate intervals. The two time gaps over 30 seconds are excluded for both methods. The classifier has 624 failed transitions and the decoder has 26. Those remaining failures are retained as separate sections.

Arc-sequence similarity is one minus normalized Levenshtein distance against the reference arc sequence, after removing consecutive duplicate arc IDs. Both methods use their observation-level predictions for this metric. Roads inserted between decoded observations are available in the route JSON. Viterbi returns a MAP sequence, so a calibrated decoder log loss is not reported.

Nine student-test trips improved. Trip 00000091 lost one correct observation, with accuracy changing from 91.5332% to 91.4188%, while continuity improved. Overall, 427 additional observations matched the inferred reference labels. The exact-label and sequence metrics inherit the original reference alignment's ambiguities.

`results/decoder/config.json` stores the selected parameters and classifier fingerprint. `validation.json` contains every decoder configuration and fold result. `comparison.json` and `per_trip.csv` contain final metrics. Each trip has saved candidate predictions, break records, and a route JSON. Maps for 00000077 (largest accuracy gain) and 00000091 (small accuracy loss) include GPS, candidate roads, reference roads, classifier predictions, connected decoder sections, corrected points, and new decoder errors. Map preview in the in-app browser was blocked by its local-file policy. Map layer names, coordinates, and route connections were checked programmatically.

## Wider candidate search

The current matcher searches the 12 nearest directed segments for each reading. Gradient Boosting was refitted on the wider sets using the same 31 geometry and trajectory features. The selected HMM transition distance scale is 5 metres and its transition strength is 0.3.

To reproduce the study from the enhanced cache, run:

```sh
python refine_experiment.py .cache/enhanced
```

The candidate audit measured coverage at widths of 5, 8, 12, 16, and 24 using the original inferred labels. Four classifier configurations and nine decoder settings per classifier were compared on the same three trip-grouped training folds. The extra features measured distance after GPS smoothing, approach and departure heading, and support from adjacent readings connected by roads. One configuration also weighted readings by the inverse square root of the training trip's length. Mean validation Top-1 accuracy selected the final configuration before student-test evaluation. Accuracy supports direct comparisons across candidate widths.

| Pipeline | Training cross-validation accuracy |
|---|---:|
| Previous five-candidate model and decoder | 81.46% |
| Five candidates with extra context | 81.55% |
| Twelve candidates with the existing features | **82.79%** |
| Twelve candidates with extra context | 82.72% |
| Twelve candidates with extra context and trip weighting | 82.77% |

| Student-test metric | Previous | Current |
|---|---:|---:|
| Top-1 accuracy after decoding | 92.61% | **94.01%** |
| Reference included in candidate set | 98.30% | **99.88%** |
| Route continuity | 99.86% | **99.97%** |
| Mean arc-sequence similarity | 0.7812 | **0.8437** |

The wider search added 268 correct GPS matches across 19,160 readings. Nine trips improved and one tied. Continuity failures fell from 26 to 6 across the same 19,148 eligible transitions. Results use the same inferred reference labels and routing limits. Two difficult training trips account for most candidate misses even at a width of 24.

`results/refinement/model.joblib` bundles the classifier and its decoder configuration. `decoder.match_trip(trip)` loads the bundle by default. Passing the earlier configuration from `results/decoder/config.json` reproduces the five-candidate comparison. Both routes were verified on all ten student-test trips using GPS and road files alone. One local pass over the ten trips, including file and model loading, took about 5.0 seconds with the current matcher and 1.6 seconds with the previous matcher.

`results/refinement/validation.json` contains the fold comparisons and decoder settings. `comparison.json` contains the aggregate and per-trip student-test metrics. The candidate audit JSON files contain coverage at each tested width and GPS-to-road distances by trip.
