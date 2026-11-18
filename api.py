import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from time import perf_counter

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from threadpoolctl import threadpool_limits

from refine import refined_route
from roads import ROOT, load_trip, project, reference_labels

BASE = Path(__file__).resolve().parent
logger = logging.getLogger("uvicorn.error")
inference_lock = Lock()


@asynccontextmanager
async def lifespan(app):
    split = Path(os.environ.get("PATHSNAP_TRIPS", ROOT / "student_test"))
    if not split.is_dir():
        raise RuntimeError("Set PATHSNAP_TRIPS to the directory containing the numbered trip folders.")
    app.state.trips = {folder.name: folder for folder in sorted(split.iterdir())
                       if folder.is_dir() and re.fullmatch(r"[0-9]{8}", folder.name)
                       and all((folder / f"{folder.name}.{ext}").is_file() for ext in ("track", "nodes", "arcs"))}
    if not app.state.trips:
        raise RuntimeError("No complete trip inputs found in PATHSNAP_TRIPS.")
    app.state.bundle = joblib.load(BASE / "results/refinement/model.joblib")
    yield


app = FastAPI(title="PathSnap", lifespan=lifespan)


@app.get("/api/trips")
def list_trips():
    return {"trips": list(app.state.trips)}


def agreement(folder, data, decoded):
    reference_file = folder / f"{folder.name}.route"
    if not reference_file.is_file():
        return None
    route = np.loadtxt(reference_file, dtype=int, ndmin=1)
    nodes, arcs, track = data
    if not len(route) or route.ndim != 1 or np.any((route < 0) | (route >= len(arcs))):
        raise ValueError("Invalid reference arc IDs")
    points = project(track[:, :2], track[0, :2])
    xy = project(nodes, track[0, :2])
    starts, ends = xy[arcs[:, 0]], xy[arcs[:, 1]]
    labels = reference_labels(route, (points, starts, ends - starts))
    return int(np.count_nonzero(labels == decoded)) / len(labels)


@app.post("/api/predict/{trip_id}")
def predict(trip_id: str):
    folder = app.state.trips.get(trip_id)
    if folder is None:
        raise HTTPException(404, "Unknown trip")
    with inference_lock, threadpool_limits(limits=2):
        start = perf_counter()
        data = load_trip(folder)
        bundle = app.state.bundle
        result = refined_route(folder, bundle["classifier"], bundle["configuration"], trip_data=data)
        seconds = perf_counter() - start
        try:
            accuracy = agreement(folder, data, result["decoded_arc_ids"])
        except (ValueError, OSError):
            logger.exception("Reference scoring unavailable for %s", trip_id)
            accuracy = None
    logger.info("trip=%s observations=%d sections=%d inference_s=%.3f accuracy=%s",
                trip_id, len(data[2]), len(result["sections"]), seconds, accuracy)
    nodes, arcs, track = data
    gps = track[:, :2]
    segments = nodes[arcs]
    near = ((segments.max(axis=1) >= gps.min(axis=0) - .001) &
            (segments.min(axis=1) <= gps.max(axis=0) + .001)).all(axis=1)
    context = segments[near]
    context = context[::max(1, (len(context) + 4999) // 5000)]
    return {"trip": trip_id, "gps": gps.tolist(), "roads": context.tolist(), "sections": result["sections"],
            "decoded_arc_ids": result["decoded_arc_ids"], "breaks": result["breaks"],
            "accuracy": accuracy, "inference_seconds": seconds}


app.mount("/", StaticFiles(directory=BASE / "ui", html=True), name="workspace")
