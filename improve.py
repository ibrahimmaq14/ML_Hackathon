"""Compare richer geometry and temporal features with fixed candidates and labels."""
import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import nn

from compare import RESULTS, SEED, measure, read
from roads import ROOT, candidates, distances, load_trip, project


from features import extended_features


def prepare(cache, output):
    for split in ["train", "student_test"]:
        dest = output / split
        dest.mkdir(parents=True, exist_ok=True)
        for source in sorted((cache / split).glob("*.npz")):
            with np.load(source) as z:
                original = np.concatenate([z["x"], z["neighbors"]], axis=-1)
                features, names = extended_features(ROOT / split / source.stem, z["ids"], original)
                np.savez_compressed(dest / source.name, x=features, neighbors=z["neighbors"],
                                    y=z["y"], ids=z["ids"], ends=z["ends"])
    return names


class ContextSage(nn.Module):
    def __init__(self, feature_count, graph):
        super().__init__()
        self.graph = graph
        self.road = nn.Linear(15 if graph else 5, 32)
        self.head = nn.Sequential(nn.Linear(feature_count - 5 + 32, 64), nn.ReLU(),
                                  nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, features):
        base = features[:, :, :-10]
        road = torch.cat([base[:, :, 4:9], features[:, :, -10:]], dim=-1) if self.graph else base[:, :, 4:9]
        embedding = torch.relu(self.road(road))
        observation = torch.cat([base[:, :, :4], base[:, :, 9:]], dim=-1)
        return self.head(torch.cat([observation, embedding], dim=-1)).squeeze(-1)


def fit_neural(x, y, graph):
    torch.manual_seed(SEED)
    # Reference-absent rows stay in evaluation, and have no candidate target to train.
    covered = y.any(axis=1)
    scale = StandardScaler().fit(x[covered].reshape(-1, x.shape[-1]))
    tensor = torch.from_numpy(scale.transform(x[covered].reshape(-1, x.shape[-1])).reshape(x[covered].shape))
    target = torch.from_numpy(y[covered].argmax(1))
    model = ContextSage(x.shape[-1]-10, graph)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    for epoch in range(35):
        for idx in torch.randperm(len(target)).split(2048):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(tensor[idx]), target[idx])
            loss.backward()
            optimizer.step()
    model.eval()
    return scale, model


def neural_scores(bundle, x):
    scale, model = bundle
    with torch.no_grad():
        flat = scale.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        return softmax(np.concatenate([model(chunk).numpy() for chunk in torch.from_numpy(flat).split(4096)]), axis=1)


def experiment(cache):
    x, y, groups = read(cache / "train")
    # read() appends graph-neighbor features after the enhanced features.
    folds = list(GroupKFold(3).split(x, groups=groups))
    configs = ["Original LR full data", "Geometry LR", "Geometry boosting", "Context MLP", "Context GraphSAGE"]
    rows = []
    oof = {}
    torch.set_num_threads(2)
    for name in configs:
        prediction = np.empty(y.shape, dtype="f")
        for fold, (train, valid) in enumerate(folds, 1):
            start = time.perf_counter()
            if name in ["Context MLP", "Context GraphSAGE"]:
                bundle = fit_neural(x[train], y[train], name == "Context GraphSAGE")
                p = neural_scores(bundle, x[valid])
            else:
                count = 9 if name == "Original LR full data" else x.shape[-1]-10
                a = x[train, :, :count].reshape(-1, count)
                b = x[valid, :, :count].reshape(-1, count)
                clf = (HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=15,
                       l2_regularization=5, early_stopping=False, random_state=SEED)
                       if name == "Geometry boosting" else
                       make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=400)))
                clf.fit(a, y[train].ravel())
                p = clf.predict_proba(b)[:, 1].reshape(y[valid].shape)
            prediction[valid] = p
            rows.append(dict(model=name, split=f"cv{fold}", **measure(y[valid], p),
                             seconds=time.perf_counter()-start))
            print(name, fold, rows[-1], flush=True)
        oof[name] = prediction
        with (RESULTS / "improvement_cv.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    # Lock selection using mean CV Macro F1 before accessing student-test features.
    means = {name: np.mean([r["macro_f1"] for r in rows if r["model"] == name]) for name in configs}
    best = max(means, key=means.get)
    record = {"selection_metric": "mean trip-grouped CV candidate Macro F1", "cv_macro_f1": means,
              "chosen_model": best, "seed": SEED, "folds": 3, "sampling": "all training observations"}
    (RESULTS / "improvement_selection.json").write_text(json.dumps(record, indent=2))
    print("Selected", best, flush=True)
    test_x, test_y, test_groups = read(cache / "student_test")
    if best in ["Context MLP", "Context GraphSAGE"]:
        bundle = fit_neural(x, y, best == "Context GraphSAGE")
        p = neural_scores(bundle, test_x)
        scale, model = bundle
        torch.save({"state_dict": model.state_dict(), "features": x.shape[-1]-10,
                    "graph": best == "Context GraphSAGE"}, RESULTS / "improved_model.pt")
        joblib.dump(scale, RESULTS / "improved_scaler.joblib")
    else:
        count = 9 if best == "Original LR full data" else x.shape[-1]-10
        clf = (HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=15, l2_regularization=5,
               early_stopping=False, random_state=SEED) if best == "Geometry boosting"
               else make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=400)))
        clf.fit(x[:, :, :count].reshape(-1, count), y.ravel())
        p = clf.predict_proba(test_x[:, :, :count].reshape(-1, count))[:, 1].reshape(test_y.shape)
        joblib.dump(clf, RESULTS / "improved_model.joblib")
    record["student_test"] = {k: float(v) for k, v in measure(test_y, p).items()}
    per_trip = []
    for trip in np.unique(test_groups):
        mask = test_groups == trip
        per_trip.append(dict(trip=f"{trip:08}", observations=int(mask.sum()),
                             **{k:float(v) for k,v in measure(test_y[mask], p[mask]).items()}))
    record["student_test_per_trip"] = per_trip
    (RESULTS / "improvement_selection.json").write_text(json.dumps(record, indent=2))
    np.savez_compressed(RESULTS / "improvement_predictions.npz", probabilities=p, groups=test_groups)
    print(json.dumps(record, indent=2), flush=True)


def score_candidates(trip):
    """Return candidate arc IDs and scores using GPS and map files only."""
    tuned_path = RESULTS / "tuning" / "selected_model.joblib"
    name = "Geometry boosting"
    if not tuned_path.exists():
        selection = json.loads((RESULTS / "improvement_selection.json").read_text())
        name = selection["chosen_model"]
    nodes, arcs, track = load_trip(trip)
    ids, original, neighbors, _ = candidates(nodes, arcs, track)
    features, _ = extended_features(trip, ids, original)
    if tuned_path.exists():
        model = joblib.load(tuned_path)
        scores = model.predict_proba(features.reshape(-1, features.shape[-1]))[:, 1].reshape(ids.shape)
    elif name in ["Context MLP", "Context GraphSAGE"]:
        state = torch.load(RESULTS / "improved_model.pt", weights_only=True)
        model = ContextSage(state["features"], state["graph"])
        model.load_state_dict(state["state_dict"])
        model.eval()
        scale = joblib.load(RESULTS / "improved_scaler.joblib")
        scores = neural_scores((scale, model), np.concatenate([features, neighbors], axis=-1))
    else:
        count = 9 if name == "Original LR full data" else features.shape[-1]
        model = joblib.load(RESULTS / "improved_model.joblib")
        scores = model.predict_proba(features[:, :, :count].reshape(-1, count))[:, 1].reshape(ids.shape)
    return ids, scores


def predict_trip(trip):
    """Score a completed trip using GPS and map files only."""
    ids, scores = score_candidates(trip)
    return ids[np.arange(len(ids)), scores.argmax(axis=1)], scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--prepare-from", type=Path)
    args = parser.parse_args()
    if args.prepare_from:
        prepare(args.prepare_from, args.cache)
    with threadpool_limits(limits=2):
        experiment(args.cache)
