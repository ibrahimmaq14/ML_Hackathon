"""Select decoder settings on out-of-fold training scores, then evaluate trips."""
import argparse
import csv
import fcntl
import hashlib
import itertools
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from compare import RESULTS, read
from decoder import POLICY, RoadNetwork, chosen_arcs, transition_geometry, viterbi
from roads import ROOT, load_trip
from tune import estimator, write_json

OUTPUT = RESULTS / "decoder"


def atomic_npz(path, **values):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **values)
    temporary.replace(path)


def out_of_fold(cache, work):
    selection = json.loads((RESULTS / "tuning" / "selection.json").read_text())
    study = json.loads((RESULTS / "tuning" / "study.json").read_text())
    manifest = {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                for file in sorted((cache / "train").glob("*.npz"))}
    if manifest != study["specification"]["training_manifest"]:
        raise ValueError("The training cache differs from the tuned model's training data")
    data, labels, groups = read(cache / "train")
    data = data[:, :, :-10]
    path = work / "out_of_fold.npz"
    metadata = work / "out_of_fold.json"
    if path.exists() and metadata.exists() and json.loads(metadata.read_text()) == selection:
        with np.load(path) as saved:
            np.testing.assert_array_equal(saved["groups"], groups)
            return saved["probabilities"].copy(), labels, groups
    scores = np.empty(labels.shape)
    for fold, (train, valid) in enumerate(GroupKFold(3).split(data, groups=groups), 1):
        assert set(groups[train]).isdisjoint(groups[valid])
        model = estimator(selection["parameters"])
        model.fit(data[train].reshape(-1, data.shape[-1]), labels[train].ravel())
        scores[valid] = model.predict_proba(data[valid].reshape(-1, data.shape[-1]))[:, 1].reshape(labels[valid].shape)
        print(f"Out-of-fold classifier fit {fold}/3 complete", flush=True)
    atomic_npz(path, probabilities=scores, groups=groups)
    write_json(metadata, selection)
    return scores, labels, groups


def cached_geometry(trip, ids, work):
    path = work / trip.parent.name / f"{trip.name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(ids.tobytes() + json.dumps(POLICY, sort_keys=True).encode())
    for ext in ["nodes", "arcs", "track"]:
        digest.update((trip / f"{trip.name}.{ext}").read_bytes())
    fingerprint = digest.hexdigest()
    if path.exists():
        with np.load(path) as saved:
            if str(saved["fingerprint"]) == fingerprint:
                return {name: saved[name].copy() for name in ["distances", "displacement", "eligible", "positions"]}
    nodes, arcs, track = load_trip(trip)
    network = RoadNetwork(nodes, arcs, track[0, :2])
    geometry = transition_geometry(network, track, ids)
    atomic_npz(path, fingerprint=fingerprint, **geometry)
    return geometry


def point_metrics(labels, states):
    predicted = np.zeros(labels.shape, dtype=bool)
    valid = states >= 0
    predicted[np.flatnonzero(valid), states[valid]] = True
    return dict(top1=float((predicted & labels).any(1).mean()),
                macro_f1=float(f1_score(labels.ravel(), predicted.ravel(), average="macro", zero_division=0)))


def continuity(geometry, states):
    pairs = np.flatnonzero(geometry["eligible"])
    available = (states[pairs] >= 0) & (states[pairs+1] >= 0)
    connected = np.zeros(len(pairs), dtype=bool)
    rows = pairs[available]
    connected[available] = np.isfinite(geometry["distances"][rows, states[rows], states[rows+1]])
    return dict(eligible_transitions=len(pairs), connected_transitions=int(connected.sum()),
                route_continuity=float(connected.mean()) if len(pairs) else None,
                excluded_time_gaps=int((~geometry["eligible"]).sum()))


def arc_sequence(arcs):
    result = []
    for arc in arcs:
        if arc >= 0 and (not result or result[-1] != arc):
            result.append(int(arc))
    return result


def edit_similarity(first, second):
    """One minus normalized Levenshtein distance on chronological arc IDs."""
    if not first and not second:
        return 1.0
    previous = list(range(len(second)+1))
    for i, left in enumerate(first, 1):
        current = [i]
        for j, right in enumerate(second, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(left != right)))
        previous = current
    return 1 - previous[-1] / max(len(first), len(second))


def select_decoder(cache, work):
    probabilities, labels, groups = out_of_fold(cache, work)
    trips = []
    for file in sorted((cache / "train").glob("*.npz")):
        mask = groups == int(file.stem)
        with np.load(file) as saved:
            ids = saved["ids"].copy()
        trip = ROOT / "train" / file.stem
        geometry = cached_geometry(trip, ids, work)
        trips.append((ids, geometry, probabilities[mask]))
        print(f"Training road transitions ready for {file.stem}", flush=True)
    configurations = list(itertools.product([10.0, 30.0, 100.0], [0.1, 0.3, 1.0, 3.0]))
    fold_masks = [valid for _, valid in GroupKFold(3).split(labels, groups=groups)]
    records = []
    classifier_states = probabilities.argmax(1)
    baseline = [point_metrics(labels[mask], classifier_states[mask]) for mask in fold_masks]
    selected = json.loads((RESULTS / "tuning" / "selection.json").read_text())
    actual = np.mean([fold["macro_f1"] for fold in baseline])
    if not np.isclose(actual, selected["cv_macro_f1"], atol=1e-12, rtol=0):
        raise ValueError("The out-of-fold classifier scores do not reproduce the tuning result")
    for beta, strength in configurations:
        start = time.perf_counter()
        decoded = [viterbi(scores, ids, geometry, beta, strength)[0]
                   for ids, geometry, scores in trips]
        states = np.concatenate(decoded)
        folds = [point_metrics(labels[mask], states[mask]) for mask in fold_masks]
        record = dict(beta_m=beta, strength=strength, folds=folds,
                      cv_macro_f1=float(np.mean([fold["macro_f1"] for fold in folds])),
                      cv_top1=float(np.mean([fold["top1"] for fold in folds])),
                      decode_seconds=time.perf_counter()-start)
        records.append(record)
        write_json(OUTPUT / "validation.json", dict(classifier_folds=baseline, configurations=records))
        print(f"Decoder beta {beta:g}, strength {strength:g}: CV F1 {record['cv_macro_f1']:.6f}, "
              f"accuracy {record['cv_top1']:.4%}", flush=True)
    best = max(records, key=lambda record: record["cv_macro_f1"])
    config = dict(beta_m=best["beta_m"], strength=best["strength"], policy=POLICY,
                  selection_metric="mean of three trip-grouped CV candidate Macro F1 scores",
                  classifier_cv_macro_f1=float(np.mean([fold["macro_f1"] for fold in baseline])),
                  decoder_cv_macro_f1=best["cv_macro_f1"], decoder_cv_top1=best["cv_top1"],
                  classifier_model_sha256=hashlib.sha256((RESULTS / "tuning" / "selected_model.joblib").read_bytes()).hexdigest())
    write_json(OUTPUT / "config.json", config)
    return config


def map_trip(trip, ids, labels, result, destination):
    import folium
    nodes, arcs, track = load_trip(trip)
    reference = np.loadtxt(trip / f"{trip.name}.route", dtype=int, ndmin=1)
    view = folium.Map(prefer_canvas=True)
    for name, selected, color, width, show in [
        ("Candidate roads", np.unique(ids[ids >= 0]), "#94a3b8", 2, False),
        ("Reference route", reference, "#15803d", 7, True),
        ("Classifier predictions", arc_sequence(result["classifier_arc_ids"]), "#ea580c", 4, True),
    ]:
        layer = folium.FeatureGroup(name=name, show=show).add_to(view)
        folium.PolyLine(nodes[arcs[selected]][:, :, ::-1].tolist(), color=color,
                        weight=width, opacity=.7).add_to(layer)
    layer = folium.FeatureGroup(name="HMM/Viterbi route").add_to(view)
    for number, section in enumerate(result["sections"], 1):
        coordinates = np.array(section["coordinates"])
        if len(coordinates) > 1:
            folium.PolyLine(coordinates[:, ::-1].tolist(), color="#7e22ce", weight=3,
                            tooltip=f"Route section {number}").add_to(layer)
    gps = folium.FeatureGroup(name="GPS observations").add_to(view)
    for lon, lat, timestamp in track:
        folium.CircleMarker([lat, lon], radius=2, color="#0369a1", weight=1,
                            tooltip=f"{timestamp:g} s").add_to(gps)
    reference_ids = np.where(labels.any(1), ids[np.arange(len(ids)), labels.argmax(1)], -1)
    covered = labels.any(1)
    classifier_correct = covered & (np.array(result["classifier_arc_ids"]) == reference_ids)
    decoder_correct = covered & (np.array(result["decoded_arc_ids"]) == reference_ids)
    for name, mask, color in [
        ("Decoder corrections", decoder_correct & ~classifier_correct, "#0f766e"),
        ("New decoder errors", classifier_correct & ~decoder_correct, "#dc2626"),
    ]:
        layer = folium.FeatureGroup(name=name).add_to(view)
        for row in np.flatnonzero(mask):
            lon, lat, timestamp = track[row]
            popup = (f"Reading {row} at {timestamp:g} s. Reference {reference_ids[row]}, "
                     f"classifier {result['classifier_arc_ids'][row]}, "
                     f"decoder {result['decoded_arc_ids'][row]}.")
            folium.CircleMarker([lat, lon], radius=4, color=color, popup=popup).add_to(layer)
    view.fit_bounds([track[:, :2].min(0)[::-1].tolist(), track[:, :2].max(0)[::-1].tolist()])
    folium.LayerControl(collapsed=False).add_to(view)
    view.save(destination)


def evaluate(cache, work, config):
    from decoder import match_trip
    model = joblib.load(RESULTS / "tuning" / "selected_model.joblib")
    observations = dict(classifier=[], decoder=[])
    predictions = dict(classifier=[], decoder=[])
    rows = []
    for file in sorted((cache / "student_test").glob("*.npz")):
        trip = ROOT / "student_test" / file.stem
        with np.load(file) as saved:
            ids, labels, features = saved["ids"], saved["y"], saved["x"]
        scores = model.predict_proba(features.reshape(-1, features.shape[-1]))[:, 1].reshape(labels.shape)
        geometry = cached_geometry(trip, ids, work)
        classifier = scores.argmax(1)
        start = time.perf_counter()
        decoded, breaks = viterbi(scores, ids, geometry, config["beta_m"], config["strength"])
        elapsed = time.perf_counter()-start
        reference = arc_sequence(np.loadtxt(trip / f"{trip.name}.route", dtype=int, ndmin=1))
        for name, states in [("classifier", classifier), ("decoder", decoded)]:
            metrics = point_metrics(labels, states)
            metrics.update(continuity(geometry, states))
            arcs = chosen_arcs(ids, states)
            metrics["arc_sequence_similarity"] = edit_similarity(reference, arc_sequence(arcs))
            rows.append(dict(trip=file.stem, model=name, observations=len(labels), **metrics))
            observations[name].append(labels)
            predictions[name].append(states)
        atomic_npz(OUTPUT / f"{file.stem}.npz", candidate_ids=ids, classifier_states=classifier,
                   decoder_states=decoded, classifier_scores=scores)
        start = time.perf_counter()
        result = match_trip(trip, config)
        inference_seconds = time.perf_counter()-start
        np.testing.assert_array_equal(result["decoded_arc_ids"], chosen_arcs(ids, decoded))
        np.testing.assert_array_equal(result["classifier_arc_ids"], chosen_arcs(ids, classifier))
        write_json(OUTPUT / f"{file.stem}_route.json", result)
        write_json(OUTPUT / f"{file.stem}_breaks.json", dict(
            breaks=breaks, decode_seconds=elapsed, inference_seconds=inference_seconds))
        print(f"Student-test {file.stem}: {rows[-2]['top1']:.2%} -> {rows[-1]['top1']:.2%}", flush=True)
    summary = {}
    for name in observations:
        labels = np.concatenate(observations[name])
        states = np.concatenate(predictions[name])
        selected = [row for row in rows if row["model"] == name]
        eligible = sum(row["eligible_transitions"] for row in selected)
        connected = sum(row["connected_transitions"] for row in selected)
        summary[name] = dict(**point_metrics(labels, states), observations=len(labels),
                             eligible_transitions=eligible, connected_transitions=connected,
                             route_continuity=connected/eligible,
                             excluded_time_gaps=sum(row["excluded_time_gaps"] for row in selected),
                             mean_trip_accuracy=float(np.mean([row["top1"] for row in selected])),
                             mean_arc_sequence_similarity=float(np.mean([row["arc_sequence_similarity"] for row in selected])))
    with (OUTPUT / "per_trip.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    write_json(OUTPUT / "comparison.json", dict(config=config, student_test=summary, per_trip=rows))
    # Select illustration trips after evaluation, with decoder settings already fixed.
    gains = {row["trip"]: row["top1"] for row in rows if row["model"] == "decoder"}
    for row in rows:
        if row["model"] == "classifier":
            gains[row["trip"]] -= row["top1"]
    for trip_id in dict.fromkeys([max(gains, key=gains.get), min(gains, key=gains.get)]):
        trip = ROOT / "student_test" / trip_id
        result = json.loads((OUTPUT / f"{trip_id}_route.json").read_text())
        with np.load(cache / "student_test" / f"{trip_id}.npz") as saved:
            map_trip(trip, saved["ids"], saved["y"], result, OUTPUT / f"{trip_id}_map.html")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    work = args.cache.parent / "decoder"
    work.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / ".experiment.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another decoder experiment is already writing these results")
        with threadpool_limits(limits=2):
            configuration = select_decoder(args.cache, work)
            evaluate(args.cache, work, configuration)
