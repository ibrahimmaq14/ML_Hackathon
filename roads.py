"""Trip geometry, candidate generation, and independent reference alignment."""
from pathlib import Path
import os
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(os.environ.get("PATHSNAP_DATA", Path.home() / "Downloads" / "Student folder"))
FEATURES = ["log_distance", "heading_cos", "speed", "heading_available", "log_length",
            "east", "north", "log_in_degree", "log_out_degree"]


def load_trip(trip):
    def read(ext, dtype):
        return np.loadtxt(trip / f"{trip.name}.{ext}", dtype=dtype, ndmin=2)
    return read("nodes", float), read("arcs", int), read("track", float)


def project(coords, origin):
    # Local tangent-plane approximation around the trip, in metres.
    delta = coords - origin
    delta[:, 0] = (delta[:, 0] + 180) % 360 - 180
    return np.radians(delta) * 6371008.8 * [np.cos(np.radians(origin[1])), 1.]


def distances(points, starts, vectors):
    denom = (vectors * vectors).sum(axis=-1)
    offsets = points - starts
    along = np.clip((offsets * vectors).sum(axis=-1) / np.maximum(denom, 1e-12), 0, 1)
    return np.linalg.norm(offsets - along[..., None] * vectors, axis=-1)


def nearest_segments(points, starts, vectors, k):
    # Index short pieces, then verify distances to the original segments.
    if not 1 <= k <= len(starts):
        raise ValueError("The candidate count must be between one and the number of road segments")
    lengths = np.linalg.norm(vectors, axis=1)
    parts = np.maximum(1, np.ceil(lengths / 60).astype(int))
    ids = np.repeat(np.arange(len(parts)), parts)
    fractions = (np.arange(len(ids)) - np.repeat(parts.cumsum()-parts, parts) + .5) / parts[ids]
    tree = cKDTree(starts[ids] + fractions[:, None] * vectors[ids])
    output = np.empty((len(points), k), int)
    pending = np.arange(len(points))
    window = 32
    while len(pending):
        count = min(window, len(ids))
        more = []
        for batch in np.array_split(pending, max(1, int(np.ceil(len(pending)*count/200000)))):
            radius, indexes = tree.query(points[batch], k=count)
            radius = radius.reshape(len(batch), count)
            indexes = indexes.reshape(len(batch), count)
            cand = ids[indexes]
            d = distances(points[batch, None], starts[cand], vectors[cand])
            for row, obs in enumerate(batch):
                unique, first = np.unique(cand[row], return_index=True)
                order = np.lexsort((unique, d[row, first]))[:k]
                # An unseen segment's closest piece midpoint is at most 30 m
                # from its closest point. Resolve all possible ties by arc ID.
                if count == len(ids) or (len(order) == k and d[row, first[order[-1]]] + 30 < radius[row,-1]):
                    output[obs] = unique[order]
                else:
                    more.append(obs)
        pending = np.asarray(more, dtype=int)
        window *= 2
    return output


def candidates(nodes, arcs, track, k=5, *, ids=None):
    xy = project(nodes, track[0,:2])
    obs = project(track[:,:2], track[0,:2])
    src, dst = arcs.T
    starts = xy[src]
    vec = xy[dst] - starts
    if ids is None:
        ids = nearest_segments(obs, starts, vec, k)
    length = np.linalg.norm(vec, axis=1)
    direction = vec / np.maximum(length[:, None], 1e-9)
    # Heading over a short centered window reduces stationary GPS jitter.
    idx = np.arange(len(obs))
    lo, hi = np.maximum(idx-2,0), np.minimum(idx+2,len(obs)-1)
    motion = obs[hi]-obs[lo]
    travel = np.linalg.norm(motion, axis=1)
    dt = track[hi,2]-track[lo,2]
    valid = (dt>0) & (dt<=12) & (travel>=3)
    heading = motion / np.maximum(travel[:,None],1e-9)
    speed = np.where(valid, travel/np.maximum(dt,1),0)
    incoming = np.bincount(dst, minlength=len(nodes))
    outgoing = np.bincount(src, minlength=len(nodes))
    road = np.column_stack([np.log1p(length), direction, np.log1p(incoming[src]), np.log1p(outgoing[dst])])
    dist = distances(obs[:,None], starts[ids], vec[ids])
    x = np.concatenate([np.stack([np.log1p(dist), (direction[ids]*heading[:,None]).sum(-1)*valid[:,None],
                                   np.broadcast_to(speed[:,None],ids.shape), np.broadcast_to(valid[:,None],ids.shape)],axis=-1),
                        road[ids]],axis=-1)
    # One-hop means for directed GraphSAGE on the road-segment graph.
    means = []
    for ends, at, degree in [(dst, src, incoming), (src, dst, outgoing)]:
        pooled = np.column_stack([np.bincount(ends, weights=col, minlength=len(nodes)) for col in road.T])
        means.append((pooled/np.maximum(degree[:,None],1))[at[ids]])
    return ids, x.astype('f'), np.concatenate(means,axis=-1).astype('f'), (obs, starts, vec)


def reference_labels(route, geometry):
    """Minimum total point-to-segment distance under nondecreasing route index."""
    obs, starts, vectors = geometry
    starts, vectors = starts[route], vectors[route]
    back = np.empty((len(obs),len(route)), np.int32)
    cost = np.zeros(len(route))
    indices = np.arange(len(route))
    for t, point in enumerate(obs):
        best = np.minimum.accumulate(cost)
        back[t] = np.maximum.accumulate(np.where(cost == best, indices, 0))
        cost = best + distances(point, starts, vectors)
    aligned = np.empty(len(obs),int)
    last = int(cost.argmin())
    for t in reversed(range(len(obs))):
        aligned[t] = last
        last = back[t,last]
    return route[aligned]


def prepare(trip):
    nodes, arcs, track = load_trip(trip)
    ids, x, neighbors, geometry = candidates(nodes, arcs, track)
    route = np.loadtxt(trip / f"{trip.name}.route", dtype=int, ndmin=1)
    labels = reference_labels(route, geometry)
    return ids, x, neighbors, labels, arcs


if __name__ == "__main__":
    import sys
    cache = Path(sys.argv[1])
    for split in ["train","student_test"]:
        dest = cache/split
        dest.mkdir(parents=True,exist_ok=True)
        for trip in sorted((ROOT/split).glob("[0-9]*")):
            ids,x,neighbors,labels,arcs = prepare(trip)
            np.savez_compressed(dest/f"{trip.name}.npz",x=x, neighbors=neighbors,
                                y=(ids==labels[:,None]), ids=ids, ends=arcs[ids])
            print(split,trip.name,flush=True)
