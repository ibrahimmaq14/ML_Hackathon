"""Road-distance HMM scoring, log-space Viterbi, and connected route output."""
from collections import OrderedDict
import heapq
import json

import numpy as np
from scipy.special import logsumexp

from roads import load_trip, project

POLICY = dict(max_gap_seconds=30.0, max_speed_mps=55.0,
              distance_slack_m=100.0, backwards_jitter_m=20.0)


class RoadNetwork:
    """Directed adjacency and bounded shortest paths, retaining original arc IDs."""

    def __init__(self, nodes, arcs, origin):
        self.xy = project(nodes, origin)
        self.arcs = np.asarray(arcs, dtype=int)
        if self.arcs.ndim != 2 or self.arcs.shape[1] != 2:
            raise ValueError("Road arcs must have two endpoint columns")
        if self.arcs.size == 0 or self.arcs.min() < 0 or self.arcs.max() >= len(nodes):
            raise ValueError("Road arcs must reference valid nodes")
        self.starts = self.xy[self.arcs[:, 0]]
        self.vectors = self.xy[self.arcs[:, 1]] - self.starts
        self.lengths = np.linalg.norm(self.vectors, axis=1)
        self.order = np.argsort(self.arcs[:, 0], kind="stable")
        self.offsets = np.r_[0, np.bincount(self.arcs[:, 0], minlength=len(nodes)).cumsum()]
        self.cache = OrderedDict()

    def shortest_paths(self, source, limit):
        cached = self.cache.get(source)
        if cached is not None and cached[0] >= limit:
            self.cache.move_to_end(source)
            return cached[1], cached[2]
        distance = {int(source): 0.0}
        previous_arc = {}
        queue = [(0.0, int(source))]
        while queue:
            cost, node = heapq.heappop(queue)
            if cost > distance[node]:
                continue
            for arc in self.order[self.offsets[node]:self.offsets[node+1]]:
                target = int(self.arcs[arc, 1])
                candidate = cost + self.lengths[arc]
                if candidate <= limit and candidate < distance.get(target, np.inf):
                    distance[target] = candidate
                    previous_arc[target] = int(arc)
                    heapq.heappush(queue, (candidate, target))
        self.cache[source] = (limit, distance, previous_arc)
        self.cache.move_to_end(source)
        if len(self.cache) > 20000:
            self.cache.popitem(last=False)
        return distance, previous_arc

    def connecting_arcs(self, source, target, limit):
        distance, previous = self.shortest_paths(source, limit)
        if distance.get(target, np.inf) > limit:
            return None
        path = []
        while target != source:
            arc = previous[target]
            path.append(arc)
            target = int(self.arcs[arc, 0])
        return path[::-1]


def candidate_positions(network, points, ids):
    safe = np.maximum(ids, 0)
    vectors = network.vectors[safe]
    offset = points[:, None] - network.starts[safe]
    along = (offset * vectors).sum(-1) / np.maximum((vectors * vectors).sum(-1), 1e-12)
    return np.clip(along, 0, 1)


def transition_geometry(network, track, ids, policy=POLICY):
    """Measure legal directed road distances between consecutive candidates."""
    if ids.shape[0] != len(track) or np.any(ids < -1) or np.any(ids >= len(network.arcs)):
        raise ValueError("Candidate IDs must align with track rows and use -1 for missing entries")
    points = project(track[:, :2], track[0, :2])
    positions = candidate_positions(network, points, ids)
    dt = np.diff(track[:, 2])
    eligible = np.isfinite(dt) & (dt > 0) & (dt <= policy["max_gap_seconds"])
    displacement = np.linalg.norm(np.diff(points, axis=0), axis=1)
    road_distance = np.full((max(0, len(track)-1), ids.shape[1], ids.shape[1]), np.inf)
    for t in np.flatnonzero(eligible):
        limit = policy["max_speed_mps"] * dt[t] + policy["distance_slack_m"]
        for i, first in enumerate(ids[t]):
            if first < 0:
                continue
            source = int(network.arcs[first, 1])
            distances, _ = network.shortest_paths(source, limit)
            remaining = (1 - positions[t, i]) * network.lengths[first]
            for j, second in enumerate(ids[t+1]):
                if second < 0:
                    continue
                between = distances.get(int(network.arcs[second, 0]), np.inf)
                length = remaining + between + positions[t+1, j] * network.lengths[second]
                delta = (positions[t+1, j] - positions[t, i]) * network.lengths[first]
                if first == second and delta >= -policy["backwards_jitter_m"]:
                    length = min(length, abs(delta))
                if length <= limit:
                    road_distance[t, i, j] = length
    return dict(distances=road_distance, displacement=displacement,
                eligible=eligible, positions=positions)


def viterbi(probabilities, ids, geometry, beta_m, strength):
    """Decode candidate states. Explicit resets preserve gaps and disconnections."""
    if beta_m <= 0 or strength <= 0:
        raise ValueError("Decoder beta and strength must be positive")
    probabilities = np.asarray(probabilities)
    if probabilities.shape != ids.shape:
        raise ValueError("Each candidate requires one classifier score")
    count, width = ids.shape
    if count == 0:
        return np.empty(0, dtype=int), []
    valid = (ids >= 0) & np.isfinite(probabilities) & (probabilities >= 0)
    emissions = np.full(ids.shape, -np.inf)
    emissions[valid] = np.log(np.clip(probabilities[valid], 1e-12, 1))
    normalizer = logsumexp(emissions, axis=1, keepdims=True)
    np.subtract(emissions, normalizer, out=emissions, where=np.isfinite(normalizer))
    log_transition = -np.abs(geometry["distances"] - geometry["displacement"][:, None, None]) / beta_m
    normalizer = logsumexp(log_transition, axis=2, keepdims=True)
    np.subtract(log_transition, normalizer, out=log_transition, where=np.isfinite(normalizer))
    log_transition *= strength
    back = np.full((count, width), -1, dtype=int)
    chosen = np.full(count, -1, dtype=int)
    scores = np.full(width, -np.inf)
    start = 0
    breaks = []

    def finish(end):
        state = int(scores.argmax())
        for row in range(end, start-1, -1):
            chosen[row] = state
            state = back[row, state]

    for row in range(count):
        active = np.isfinite(scores).any()
        if not valid[row].any():
            if active:
                finish(row-1)
            scores[:] = -np.inf
            start = row+1
            breaks.append(dict(observation=row, reason="missing_candidates"))
            continue
        gap = row > 0 and not geometry["eligible"][row-1]
        if gap and active:
            finish(row-1)
            scores[:] = -np.inf
            active = False
        if gap:
            breaks.append(dict(observation=row, reason="time_gap"))
        if not active:
            start = row
            scores = emissions[row].copy()
        else:
            options = scores[:, None] + log_transition[row-1]
            best = options.argmax(axis=0)
            updated = options[best, np.arange(width)] + emissions[row]
            if not np.isfinite(updated).any():
                finish(row-1)
                start = row
                scores = emissions[row].copy()
                breaks.append(dict(observation=row, reason="unreachable_candidates"))
            else:
                back[row] = best
                scores = updated
        scores -= np.max(scores)
    if np.isfinite(scores).any():
        finish(count-1)
    return chosen, breaks


def chosen_arcs(ids, states):
    result = np.full(len(states), -1, dtype=int)
    valid = states >= 0
    result[valid] = ids[np.flatnonzero(valid), states[valid]]
    return result


def route_sections(network, track, ids, geometry, states, breaks, policy=POLICY):
    """Fill intermediate roads and retain separate geometry at every reset."""
    arcs = chosen_arcs(ids, states)
    reset = {item["observation"] for item in breaks}
    sections = []
    section = None
    for row, arc in enumerate(arcs):
        if arc < 0:
            section = None
            continue
        point = network.starts[arc] + geometry["positions"][row, states[row]] * network.vectors[arc]
        if section is None or row in reset:
            section = dict(start_observation=row, end_observation=row, arc_ids=[int(arc)], xy=[point.tolist()])
            sections.append(section)
            continue
        previous = arcs[row-1]
        delta = ((geometry["positions"][row, states[row]] - geometry["positions"][row-1, states[row-1]])
                 * network.lengths[arc])
        shortcut = previous == arc and delta >= -policy["backwards_jitter_m"]
        if not shortcut:
            limit = (policy["max_speed_mps"] * (track[row, 2]-track[row-1, 2])
                     + policy["distance_slack_m"])
            bridge = network.connecting_arcs(int(network.arcs[previous, 1]),
                                             int(network.arcs[arc, 0]), limit)
            if bridge is None:
                raise RuntimeError("A decoded transition has no connecting road path")
            section["xy"].append(network.xy[network.arcs[previous, 1]].tolist())
            for intermediate in bridge:
                section["arc_ids"].append(intermediate)
                section["xy"].append(network.xy[network.arcs[intermediate, 1]].tolist())
            section["arc_ids"].append(int(arc))
        section["xy"].append(point.tolist())
        section["end_observation"] = row
    return sections


def match_trip(trip, config=None):
    """Inference entry point. Reads GPS/map files, model, and decoder settings."""
    from compare import RESULTS
    from improve import score_candidates

    if config is None:
        refined_path = RESULTS / "refinement" / "model.joblib"
        if refined_path.is_file():
            import joblib
            from refine import refined_route

            bundle = joblib.load(refined_path)
            return refined_route(trip, bundle["classifier"], bundle["configuration"])
        config = json.loads((RESULTS / "decoder" / "config.json").read_text())
    policy = config["policy"]
    nodes, arcs, track = load_trip(trip)
    ids, probabilities = score_candidates(trip)
    network = RoadNetwork(nodes, arcs, track[0, :2])
    geometry = transition_geometry(network, track, ids, policy)
    return decode_route(network, track, ids, probabilities, geometry, config)


def decode_route(network, track, ids, probabilities, geometry, config):
    policy = config["policy"]
    states, breaks = viterbi(probabilities, ids, geometry, config["beta_m"], config["strength"])
    sections = route_sections(network, track, ids, geometry, states, breaks, policy)
    # Convert the local projection back to longitude/latitude for consumers.
    scale = np.array([np.cos(np.radians(track[0, 1])), 1]) * 6371008.8
    for section in sections:
        coordinates = np.degrees(np.array(section.pop("xy")) / scale) + track[0, :2]
        coordinates[:, 0] = (coordinates[:, 0] + 180) % 360 - 180
        section["coordinates"] = coordinates.tolist()
    return dict(classifier_arc_ids=ids[np.arange(len(ids)), probabilities.argmax(1)].tolist(),
                decoded_arc_ids=chosen_arcs(ids, states).tolist(),
                sections=sections, breaks=breaks)
