import numpy as np

from roads import distances, load_trip, project


def extended_features(trip, ids, original, *, trip_data=None):
    nodes, arcs, track = load_trip(trip) if trip_data is None else trip_data
    points = project(track[:, :2], track[0, :2])
    roads = project(nodes, track[0, :2])[arcs[ids]]
    starts, ends = roads[:, :, 0], roads[:, :, 1]
    vectors = ends - starts
    length = np.linalg.norm(vectors, axis=-1)
    direction = vectors / np.maximum(length[..., None], 1e-9)
    offset = points[:, None] - starts
    position = (offset * direction).sum(-1) / np.maximum(length, 1e-9)
    distance = distances(points[:, None], starts, vectors)
    extra = [
        position.clip(-5, 6), position.clip(0, 1),
        np.log1p(np.linalg.norm(offset, axis=-1)),
        np.log1p(np.linalg.norm(points[:, None] - ends, axis=-1)),
        np.log1p(distance) - np.log1p(distance[:, :1]),
        distance / np.maximum(distance[:, :1], 2),
    ]
    names = ["position", "clipped_position", "log_start_distance", "log_end_distance",
             "relative_log_distance", "distance_ratio"]
    # Completed-trip reconstruction can use observations on both sides.
    row = np.arange(len(points))
    for window in [1, 4, 8, 16]:
        lo, hi = np.maximum(row-window, 0), np.minimum(row+window, len(points)-1)
        dt = track[hi, 2] - track[lo, 2]
        motion = points[hi] - points[lo]
        travel = np.linalg.norm(motion, axis=-1)
        valid = (dt > 0) & (dt <= 3*window) & (travel >= 3)
        heading = motion / np.maximum(travel[:, None], 1e-9)
        cosine = (heading[:, None] * direction).sum(-1) * valid[:, None]
        extra.extend([
            cosine,
            np.broadcast_to(valid[:, None], ids.shape),
            distances(points[lo, None], starts, vectors).clip(0, 1000) / 100,
            distances(points[hi, None], starts, vectors).clip(0, 1000) / 100,
        ])
        names.extend([f"heading_{window}", f"valid_{window}", f"before_distance_{window}", f"after_distance_{window}"])
    enhanced = np.concatenate([original[:, :, :9], np.stack(extra, axis=-1)], axis=-1)
    return enhanced.astype("f"), names
