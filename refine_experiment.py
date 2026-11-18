"""Audit candidate recall and evaluate refinements against the frozen decoder."""
import argparse
import fcntl
import hashlib
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from compare import RESULTS
from decoder import POLICY, chosen_arcs, viterbi
from decoder_experiment import atomic_npz, continuity, edit_similarity, arc_sequence, point_metrics
from refine import (BASE_FEATURE_COUNT, WIDTH, add_context, classifier_probabilities,
                    refined_route, trim_geometry, trip_features)
from roads import ROOT, distances, load_trip, nearest_segments, project, reference_labels
from tune import estimator, write_json

OUTPUT = RESULTS / "refinement"
CANDIDATE_AUDIT = (5, 8, 12, 16, 24)
SETTINGS = dict(
    context_5=dict(candidate_count=5, context_features=True, weight_trips=False),
    geometry_12=dict(candidate_count=WIDTH, context_features=False, weight_trips=False),
    context_12=dict(candidate_count=WIDTH, context_features=True, weight_trips=False),
    context_12_balanced=dict(candidate_count=WIDTH, context_features=True, weight_trips=True),
)
DECODER_GRID = list(itertools.product((5., 10., 25.), (.1, .3, .75)))


def prepare_trip(trip, destination):
    trip_data = load_trip(trip)
    nodes, arcs, track = trip_data
    points = project(track[:, :2], track[0, :2])
    xy = project(nodes, track[0, :2])
    starts = xy[arcs[:, 0]]
    vectors = xy[arcs[:, 1]] - starts
    wide = nearest_segments(points, starts, vectors, min(max(CANDIDATE_AUDIT), len(arcs)))
    route = np.loadtxt(trip / f"{trip.name}.route", dtype=int, ndmin=1)
    reference = reference_labels(route, (points, starts, vectors))
    ids, base, geometry, network = trip_features(trip, WIDTH, POLICY, trip_data=trip_data, ids=wide[:, :WIDTH])
    context = add_context(track, network, ids, base, geometry)
    if ids.shape[1] < WIDTH:
        raise ValueError(f"Trip {trip.name} has fewer than {WIDTH} distinct arcs")
    y = ids == reference[:, None]
    audit = dict(trip=trip.name, observations=len(ids), misses={
        str(k): int((~(wide[:, :k] == reference[:, None]).any(1)).sum()) for k in CANDIDATE_AUDIT},
        median_reference_distance_m=float(np.median(distances(points, starts[reference], vectors[reference]))),
        median_nearest_distance_m=float(np.median(distances(points, starts[wide[:, 0]], vectors[wide[:, 0]]))))
    atomic_npz(destination, x=context, y=y, ids=ids, **geometry)
    return audit


def prepare_split(split, work, baseline_cache):
    destination = work / split
    destination.mkdir(parents=True, exist_ok=True)
    audit = []
    for trip in sorted((ROOT / split).glob("[0-9]*")):
        file = destination / f"{trip.name}.npz"
        known = destination / f"{trip.name}.json"
        checksum = hashlib.sha256(json.dumps([1, WIDTH, POLICY, CANDIDATE_AUDIT], sort_keys=True).encode())
        for extension in ("nodes", "arcs", "track", "route"):
            checksum.update((trip / f"{trip.name}.{extension}").read_bytes())
        prior = baseline_cache / split / file.name
        checksum.update(prior.read_bytes())
        identity = checksum.hexdigest()
        record = json.loads(known.read_text()) if file.exists() and known.exists() else {}
        if record.get("identity") != identity:
            record = prepare_trip(trip, file)
            with np.load(prior) as old, np.load(file) as new:
                np.testing.assert_array_equal(new["ids"][:, :5], old["ids"])
                np.testing.assert_array_equal(new["y"][:, :5], old["y"])
                np.testing.assert_array_equal(new["x"][:, :5, :BASE_FEATURE_COUNT], old["x"])
            record["identity"] = identity
            write_json(known, record)
        audit.append({key: value for key, value in record.items() if key != "identity"})
        if len(audit) % 10 == 0:
            print(f"Prepared {len(audit)} {split} trips", flush=True)
    if len(audit) != {"train": 80, "student_test": 10}[split]:
        raise ValueError(f"Incomplete {split} partition")
    write_json(OUTPUT / f"{split}_candidate_audit.json", dict(
        observations=sum(r["observations"] for r in audit),
        misses={str(k): sum(r["misses"][str(k)] for r in audit) for k in CANDIDATE_AUDIT},
        trips=audit))


def read_split(path):
    x, y, groups = [], [], []
    for file in sorted(path.glob("*.npz")):
        with np.load(file) as saved:
            x.append(saved["x"])
            y.append(saved["y"])
            groups.append(np.full(len(y[-1]), int(file.stem)))
    return np.concatenate(x), np.concatenate(y), np.concatenate(groups)


def views(x, y, setting):
    width = setting["candidate_count"]
    count = x.shape[-1] if setting["context_features"] else BASE_FEATURE_COUNT
    return x[:, :width, :count], y[:, :width]


def fit_model(x, y, groups, setting, params):
    width = x.shape[1]
    weights = None
    if setting["weight_trips"]:
        _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
        per_observation = 1 / np.sqrt(counts[inverse])
        weights = np.repeat(per_observation / per_observation.mean(), width)
    model = estimator(params)
    model.fit(x.reshape(-1, x.shape[-1]), y.ravel(), sample_weight=weights)
    return model


def cv_record(labels, states, folds):
    records = [point_metrics(labels[valid], states[valid]) for _, valid in folds]
    return dict(folds=records, cv_top1=float(np.mean([r["top1"] for r in records])),
                cv_macro_f1=float(np.mean([r["macro_f1"] for r in records])))


def compare_on_training(work, features, labels, groups, params):
    folds = list(GroupKFold(3).split(labels, groups=groups))
    _, starts = np.unique(groups, return_index=True)
    stops = np.r_[starts[1:], len(groups)]
    paths = sorted((work / "train").glob("*.npz"))
    fingerprint = hashlib.sha256()
    for path in paths:
        fingerprint.update(path.name.encode())
        fingerprint.update(path.read_bytes())
    baseline = json.loads((RESULTS / "decoder" / "config.json").read_text())
    study = dict(metric="mean trip-grouped validation pointwise accuracy",
                 baseline_cv_top1=baseline["decoder_cv_top1"],
                 candidate_widths=[5, WIDTH],
                 models={})
    for name, setting in SETTINGS.items():
        x, y = views(features, labels, setting)
        width = x.shape[1]
        score_path = work / f"{name}_oof.npz"
        identity = json.dumps([params, setting, fingerprint.hexdigest()], sort_keys=True)
        scores = None
        if score_path.exists():
            with np.load(score_path) as saved:
                if "identity" in saved and str(saved["identity"]) == identity:
                    np.testing.assert_array_equal(saved["groups"], groups)
                    scores = saved["scores"]
        if scores is None:
            scores = np.empty(y.shape)
            for fold, (train, valid) in enumerate(folds, 1):
                assert set(groups[train]).isdisjoint(groups[valid])
                model = fit_model(x[train], y[train], groups[train], setting, params)
                scores[valid] = classifier_probabilities(model, x[valid])
                print(f"{name} held-out fold {fold}/3 complete", flush=True)
            atomic_npz(score_path, groups=groups, scores=scores, identity=identity)
        states = np.empty((len(DECODER_GRID), len(y)), dtype=np.int16)
        for file, start, stop in zip(paths, starts, stops, strict=True):
            assert int(file.stem) == groups[start]
            with np.load(file) as loaded:
                ids = loaded["ids"][:, :width]
                geometry = trim_geometry(loaded, width)
            for i, (beta, strength) in enumerate(DECODER_GRID):
                states[i, start:stop] = viterbi(scores[start:stop], ids, geometry, beta, strength)[0]
        choices = [dict(beta_m=beta, strength=strength, **cv_record(y, states[i], folds))
                   for i, (beta, strength) in enumerate(DECODER_GRID)]
        best = max(choices, key=lambda r: r["cv_top1"])
        study["models"][name] = dict(settings=setting, classifier=cv_record(y, scores.argmax(1), folds),
                                    best_decoder=best, decoder_trials=choices)
        write_json(OUTPUT / "validation.json", study)
        print(f"{name}: CV decoder accuracy {best['cv_top1']:.4%}", flush=True)
    name = max(study["models"], key=lambda key: study["models"][key]["best_decoder"]["cv_top1"])
    selected = study["models"][name]
    return dict(model=name, **selected["settings"], **selected["best_decoder"], policy=POLICY,
                context_candidate_count=WIDTH, classifier_parameters=params,
                baseline_cv_top1=baseline["decoder_cv_top1"])


def evaluate(work, selected, model):
    baseline = json.loads((RESULTS / "decoder" / "comparison.json").read_text())
    per_trip = []
    connected_total = eligible_total = 0
    reference_misses = 0
    for file in sorted((work / "student_test").glob("*.npz")):
        trip = ROOT / "student_test" / file.stem
        with np.load(file) as saved:
            features, labels = views(saved["x"], saved["y"], selected)
            width = selected["candidate_count"]
            ids = saved["ids"][:, :width]
            geometry = trim_geometry(saved, width)
        scores = classifier_probabilities(model, features)
        states, _ = viterbi(scores, ids, geometry, selected["beta_m"], selected["strength"])
        route = refined_route(trip, model, selected)
        np.testing.assert_array_equal(route["decoded_arc_ids"], chosen_arcs(ids, states))
        np.testing.assert_array_equal(route["classifier_arc_ids"], chosen_arcs(ids, scores.argmax(1)))
        reference = arc_sequence(np.loadtxt(trip / f"{trip.name}.route", dtype=int, ndmin=1))
        match = point_metrics(labels, states)
        link = continuity(geometry, states)
        per_trip.append(dict(trip=file.stem, observations=len(ids),
                             correct=int(round(match["top1"]*len(ids))),
                             classifier_top1=point_metrics(labels, scores.argmax(1))["top1"],
                             **match, **link, arc_sequence_similarity=edit_similarity(
                                 reference, arc_sequence(route["decoded_arc_ids"]))))
        connected_total += link["connected_transitions"]
        eligible_total += link["eligible_transitions"]
        reference_misses += int((~labels.any(1)).sum())
        atomic_npz(OUTPUT / f"{trip.name}.npz", decoder_arc_ids=np.array(route["decoded_arc_ids"]),
                   classifier_arc_ids=np.array(route["classifier_arc_ids"]))
        print(f"Student-test {trip.name}: {match['top1']:.2%}", flush=True)
    observations = sum(row["observations"] for row in per_trip)
    old = baseline["student_test"]["decoder"]
    old_trips = {row["trip"]: row for row in baseline["per_trip"] if row["model"] == "decoder"}
    for row in per_trip:
        row["previous_top1"] = old_trips[row["trip"]]["top1"]
    refined = dict(top1=sum(row["correct"] for row in per_trip)/observations,
                   route_continuity=connected_total/eligible_total,
                   connected_transitions=connected_total, eligible_transitions=eligible_total,
                   mean_trip_accuracy=float(np.mean([r["top1"] for r in per_trip])),
                   mean_arc_sequence_similarity=float(np.mean([r["arc_sequence_similarity"] for r in per_trip])),
                   candidate_recall=1-reference_misses/observations,
                   observations=observations)
    report = dict(selection=selected, previous=old, refined=refined, per_trip=per_trip)
    write_json(OUTPUT / "comparison.json", report)
    return report


def save_model(model, selected):
    temporary = OUTPUT / "model.joblib.tmp"
    joblib.dump(dict(classifier=model, configuration=selected), temporary)
    temporary.replace(OUTPUT / "model.joblib")


def run(cache, work):
    study = json.loads((RESULTS / "tuning" / "selection.json").read_text())
    params = study["parameters"]
    prepare_split("train", work, cache)
    features, labels, groups = read_split(work / "train")
    selected = compare_on_training(work, features, labels, groups, params)
    # Persist the locked training choice before loading the student-test partition.
    write_json(OUTPUT / "selection.json", selected)
    x, y = views(features, labels, selected)
    model = fit_model(x, y, groups, selected, params)
    del x, y, features, labels, groups
    prepare_split("student_test", work, cache)
    report = evaluate(work, selected, model)
    if selected["cv_top1"] > selected["baseline_cv_top1"]:
        save_model(model, selected)
    print(json.dumps(report["refined"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_cache", type=Path)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / ".experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with threadpool_limits(limits=2):
            run(args.baseline_cache, args.baseline_cache.parent / "refinement")
