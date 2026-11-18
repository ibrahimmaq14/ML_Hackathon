"""Candidate and trajectory features for the route-matching refinement."""
import numpy as np

from decoder import RoadNetwork, decode_route, transition_geometry
from improve import extended_features
from roads import candidates, distances, load_trip, project

WIDTH = 12
BASE_FEATURE_COUNT = 31


def trajectory_features(track, network, ids, geometry):
    """Features use the observed path and the road graph, with no reference labels."""
    points = project(track[:, :2], track[0, :2])
    timestamps = track[:, 2]
    starts = network.starts[ids]
    vectors = network.vectors[ids]
    unit = vectors / np.maximum(network.lengths[ids, None], 1e-9)
    projected = starts + geometry["positions"][..., None] * vectors
    raw_distance = np.linalg.norm(points[:, None] - projected, axis=-1)
    row = np.arange(len(points))
    features = []

    for window in (2, 6):
        offsets = np.clip(row[:, None]+np.arange(-window, window+1), 0, len(row)-1)
        elapsed = timestamps[offsets] - timestamps[:, None]
        valid = np.abs(elapsed) <= 3*window
        weight = valid / valid.sum(1, keepdims=True)
        center = (points[offsets]*weight[..., None]).sum(1)
        features.append(np.log1p(distances(center[:, None], starts, vectors)))
        for direction in (-1, 1):
            other = np.clip(row + direction*window, 0, len(row)-1)
            elapsed = np.abs(timestamps[other]-timestamps)
            delta = direction * (points[other]-points)
            displacement = np.linalg.norm(delta, axis=1)
            heading = delta / np.maximum(displacement[:, None], 1e-9)
            available = (elapsed > 0) & (elapsed <= 3*window) & (displacement >= 2)
            cosine = (heading[:, None]*unit).sum(-1)
            features.append(np.where(available[:, None], cosine, np.nan))

    # Score local evidence for entering and leaving each candidate independently.
    # Every neighbor penalty uses its observed distance and direction agreement.
    lo = np.maximum(row-2, 0)
    hi = np.minimum(row+2, len(row)-1)
    motion = points[hi]-points[lo]
    travel = np.linalg.norm(motion, axis=1)
    heading = motion / np.maximum(travel[:, None], 1e-9)
    alignment = (unit*heading[:, None]).sum(-1)
    neighbor_cost = raw_distance + np.where(travel[:, None] >= 3, 12*(1-alignment), 0)
    error = np.abs(geometry["distances"]-geometry["displacement"][:, None, None])
    eligible = geometry["eligible"]
    for axis, evidence, pad in [
        (2, neighbor_cost[1:, None, :], ((0, 1), (0, 0))),
        (1, neighbor_cost[:-1, :, None], ((1, 0), (0, 0))),
    ]:
        any_road = np.isfinite(error).any(axis)
        best_error = np.min(error, axis=axis)
        best_supported = np.min(error+evidence, axis=axis)
        supported_fraction = np.isfinite(error).mean(axis)
        for values in (best_error, best_supported):
            features.append(np.pad(np.where(eligible[:, None] & any_road,
                                            np.log1p(values), np.nan), pad, constant_values=np.nan))
        features.append(np.pad(np.where(eligible[:, None], supported_fraction, np.nan),
                               pad, constant_values=np.nan))
    features.append(np.broadcast_to(np.arange(ids.shape[1]), ids.shape))
    return np.stack(features, axis=-1).astype("f")


def trip_features(trip, width, policy, *, trip_data=None, ids=None):
    if trip_data is None:
        trip_data = load_trip(trip)
    nodes, arcs, track = trip_data
    ids, original, _, _ = candidates(nodes, arcs, track, min(width, len(arcs)), ids=ids)
    features, _ = extended_features(trip, ids, original, trip_data=trip_data)
    network = RoadNetwork(nodes, arcs, track[0, :2])
    geometry = transition_geometry(network, track, ids, policy)
    return ids, features, geometry, network


def add_context(track, network, ids, base, geometry):
    return np.concatenate([base, trajectory_features(track, network, ids, geometry)], axis=-1)


def classifier_probabilities(model, features):
    return model.predict_proba(features.reshape(-1, features.shape[-1]))[:, 1].reshape(features.shape[:2])


def trim_geometry(geometry, width):
    return dict(distances=geometry["distances"][:, :width, :width],
                positions=geometry["positions"][:, :width],
                displacement=geometry["displacement"], eligible=geometry["eligible"])


def refined_route(trip, model, config, *, trip_data=None):
    if trip_data is None:
        trip_data = load_trip(trip)
    width = config["candidate_count"]
    search_width = config["context_candidate_count"] if config["context_features"] else width
    ids, features, geometry, network = trip_features(
        trip, search_width, config["policy"], trip_data=trip_data)
    track = trip_data[2]
    if config["context_features"]:
        features = add_context(track, network, ids, features, geometry)
    probabilities = classifier_probabilities(model, features[:, :width])
    return decode_route(network, track, ids[:, :width], probabilities, trim_geometry(geometry, width), config)
