"""Gaussian-process Bayesian tuning with the existing trip-grouped folds."""
import argparse
import fcntl
import hashlib
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from scipy.special import ndtr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from compare import RESULTS, SEED, measure, read
from roads import FEATURES

OUTPUT = RESULTS / "tuning"
BASELINE = dict(learning_rate=0.1, max_iter=180, max_leaf_nodes=15,
                min_samples_leaf=20, l2_regularization=5.0, max_bins=255)
SEARCH_SPACE = {
    "learning_rate": {"distribution": "log-uniform", "low": 0.025, "high": 0.2},
    "max_iter": {"distribution": "discrete-uniform", "values": list(range(100, 501, 25))},
    "max_leaf_nodes": {"distribution": "categorical", "values": [7, 15, 31, 63]},
    "min_samples_leaf": {"distribution": "categorical", "values": [10, 20, 40, 80, 160]},
    "l2_regularization": {"distribution": "log-uniform", "low": 0.01, "high": 30.0},
    "max_bins": {"distribution": "categorical", "values": [63, 127, 255]},
}
FEATURE_NAMES = FEATURES + [
    "position", "clipped_position", "log_start_distance", "log_end_distance",
    "relative_log_distance", "distance_ratio",
] + [name for window in [1, 4, 8, 16] for name in [
    f"heading_{window}", f"valid_{window}", f"before_distance_{window}",
    f"after_distance_{window}",
]]


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def signature(parameters):
    return json.dumps(parameters, sort_keys=True)


def encode(parameters):
    """Represent discrete and logarithmic search dimensions on a unit scale."""
    values = []
    for name, space in SEARCH_SPACE.items():
        value = parameters[name]
        if space["distribution"] == "log-uniform":
            low, high = np.log([space["low"], space["high"]])
            values.append((np.log(value) - low) / (high - low))
        else:
            options = space["values"]
            # The original 180-iteration baseline is outside the discrete grid.
            values.append(np.interp(value, options, np.linspace(0, 1, len(options))))
    return values


def sample(rng):
    parameters = {}
    for name, space in SEARCH_SPACE.items():
        if space["distribution"] == "log-uniform":
            parameters[name] = float(np.exp(rng.uniform(np.log(space["low"]), np.log(space["high"]))))
        else:
            parameters[name] = int(rng.choice(space["values"]))
    return parameters


def propose(trials):
    rng = np.random.default_rng(SEED + len(trials))
    used = {signature(trial["parameters"]) for trial in trials}
    options = [sample(rng) for _ in range(2048)]
    options = [candidate for candidate in options if signature(candidate) not in used]
    if len(trials) < 6:
        return options[0], "random_initialisation"
    # Fit a Matérn surrogate to completed cross-validation objectives.
    kernel = ConstantKernel(1.0, (0.01, 100)) * Matern(
        length_scale=np.ones(len(SEARCH_SPACE)), length_scale_bounds=(0.05, 5.0), nu=2.5)
    surrogate = GaussianProcessRegressor(
        kernel=kernel, alpha=1e-5, normalize_y=True, n_restarts_optimizer=2,
        random_state=SEED)
    objectives = np.array([trial["cv_macro_f1"] for trial in trials])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        surrogate.fit([encode(trial["parameters"]) for trial in trials], objectives)
    mean, std = surrogate.predict([encode(candidate) for candidate in options], return_std=True)
    gain = mean - objectives.max() - 0.0002
    z = gain / np.maximum(std, 1e-12)
    expected_improvement = gain * ndtr(z) + std * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    return options[int(expected_improvement.argmax())], "gp_expected_improvement"


def estimator(parameters):
    return HistGradientBoostingClassifier(
        **parameters, early_stopping=False, random_state=SEED)


def timed_scores(model, features):
    # Warm up once, then report the median of five scoring runs.
    model.predict_proba(features)
    elapsed = []
    for _ in range(5):
        start = time.perf_counter()
        probabilities = model.predict_proba(features)[:, 1]
        elapsed.append(time.perf_counter() - start)
    return probabilities, float(np.median(elapsed))


def run(cache, total_trials):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    training_files = sorted((cache / "train").glob("*.npz"))
    if len(training_files) != 80:
        raise ValueError("Expected the complete set of 80 training trips")
    manifest = {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in training_files}
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    x, y, groups = read(cache / "train")
    # Remove the ten neighbor features appended by read().
    x = x[:, :, :-10]
    assert x.shape[-1] == len(FEATURE_NAMES)
    folds = list(GroupKFold(3).split(x, groups=groups))
    specification = {
        "schema": 1, "optimizer": "Gaussian process with Matern 2.5 kernel and expected improvement",
        "objective": "mean of three trip-grouped CV candidate Macro F1 scores",
        "initialisation": "original baseline followed by five random trials",
        "seed": SEED, "search_space": SEARCH_SPACE, "baseline": BASELINE,
        "feature_names": FEATURE_NAMES, "training_manifest": manifest,
        "training_fingerprint": fingerprint,
        "folds": [{"fold": i, "training_trips": np.unique(groups[train]).tolist(),
                   "validation_trips": np.unique(groups[valid]).tolist()}
                  for i, (train, valid) in enumerate(folds, 1)],
    }
    state_path = OUTPUT / "study.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(specification=specification, trials=[])
    if state["specification"] != specification:
        raise ValueError("The saved study uses different data, folds, or settings")
    trials = state["trials"]
    if len(trials) > total_trials:
        raise ValueError("Requested trial count is smaller than the saved study")
    write_json(state_path, state)
    # Hold fold arrays in memory to keep all trials on identical data.
    fold_data = []
    for train, valid in folds:
        assert set(groups[train]).isdisjoint(groups[valid])
        fold_data.append((x[train].reshape(-1, x.shape[-1]), y[train].ravel(),
                          x[valid].reshape(-1, x.shape[-1]), y[valid]))
    study_start = time.perf_counter()
    while len(trials) < total_trials:
        if not trials:
            parameters, source = BASELINE.copy(), "untuned_baseline"
        else:
            parameters, source = propose(trials)
        trial = dict(number=len(trials), parameters=parameters, proposal=source, folds=[])
        for fold_number, (train_x, train_y, valid_x, valid_y) in enumerate(fold_data, 1):
            model = estimator(parameters)
            start = time.perf_counter()
            model.fit(train_x, train_y)
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            probabilities = model.predict_proba(valid_x)[:, 1].reshape(valid_y.shape)
            inference_seconds = time.perf_counter() - start
            metrics = {key: float(value) for key, value in measure(valid_y, probabilities).items()}
            trial["folds"].append(dict(fold=fold_number, fit_seconds=fit_seconds,
                                       inference_seconds=inference_seconds, **metrics))
        trial["cv_macro_f1"] = float(np.mean([fold["macro_f1"] for fold in trial["folds"]]))
        trial["cv_top1"] = float(np.mean([fold["top1"] for fold in trial["folds"]]))
        trial["cv_binary_log_loss"] = float(np.mean([fold["binary_log_loss"] for fold in trial["folds"]]))
        trials.append(trial)
        write_json(state_path, state)
        print(f"Trial {len(trials)}/{total_trials}: F1 {trial['cv_macro_f1']:.6f}, "
              f"accuracy {trial['cv_top1']:.4%}, best F1 "
              f"{max(item['cv_macro_f1'] for item in trials):.6f}", flush=True)
    # Fix the winner before loading the student-test data.
    best = max(trials, key=lambda trial: trial["cv_macro_f1"])
    selection = dict(trial_number=best["number"], parameters=best["parameters"],
                     cv_macro_f1=best["cv_macro_f1"], training_fingerprint=fingerprint)
    write_json(OUTPUT / "selection.json", selection)
    test_x, test_y, test_groups = read(cache / "student_test")
    test_x = test_x[:, :, :-10]
    comparison = []
    prediction_arrays = {"groups": test_groups}
    for name, trial in [("Untuned", trials[0]), ("Selected", best)]:
        model = estimator(trial["parameters"])
        start = time.perf_counter()
        model.fit(x.reshape(-1, x.shape[-1]), y.ravel())
        fit_seconds = time.perf_counter() - start
        flat, inference_seconds = timed_scores(model, test_x.reshape(-1, test_x.shape[-1]))
        probabilities = flat.reshape(test_y.shape)
        row = dict(model=name, parameters=trial["parameters"], cv_macro_f1=trial["cv_macro_f1"],
                   cv_top1=trial["cv_top1"], cv_binary_log_loss=trial["cv_binary_log_loss"],
                   fit_seconds=fit_seconds, inference_seconds=inference_seconds,
                   student_test={key: float(value) for key, value in measure(test_y, probabilities).items()},
                   per_trip=[])
        for trip in np.unique(test_groups):
            mask = test_groups == trip
            row["per_trip"].append(dict(trip=f"{trip:08}", observations=int(mask.sum()),
                                        **{key: float(value) for key, value in measure(test_y[mask], probabilities[mask]).items()}))
        comparison.append(row)
        prediction_arrays[name.lower()] = probabilities
        if name == "Selected":
            temporary = OUTPUT / "selected_model.tmp"
            joblib.dump(model, temporary)
            temporary.replace(OUTPUT / "selected_model.joblib")
            write_json(OUTPUT / "model_metadata.json", dict(
                selection=selection, features=FEATURE_NAMES, candidate_count=int(test_y.shape[1]),
                estimator_parameters=model.get_params(), training_observations=len(y)))
    np.savez_compressed(OUTPUT / "predictions.npz", **prediction_arrays)
    write_json(OUTPUT / "comparison.json", dict(
        completed_trials=len(trials), study_and_refit_seconds=time.perf_counter()-study_start,
        observations=dict(train=len(y), student_test=len(test_y)),
        inference_timing="median of five candidate-scoring runs after one warmup, two CPU threads",
        comparison=comparison))
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--trials", type=int, default=24)
    args = parser.parse_args()
    if args.trials < 7:
        parser.error("At least seven trials are required to include a Bayesian proposal")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / ".study.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another tuning process is already writing this study")
        with threadpool_limits(limits=2):
            run(args.cache, args.trials)
