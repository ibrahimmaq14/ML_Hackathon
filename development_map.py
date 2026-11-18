"""Interactive analysis map. Development data only, never served by the evaluator API."""
import argparse
from pathlib import Path

import folium
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from refine import refined_route
from roads import ROOT, load_trip

BASE = Path(__file__).resolve().parent


def build(trip):
    data = load_trip(trip)
    nodes, arcs, track = data
    bundle = joblib.load(BASE / "results/refinement/model.joblib")
    with threadpool_limits(limits=2):
        prediction = refined_route(trip, bundle["classifier"], bundle["configuration"], trip_data=data)
    route = np.loadtxt(trip / f"{trip.name}.route", dtype=int, ndmin=1)
    gps = track[:, :2]
    south, north = gps.min(axis=0), gps.max(axis=0)
    road_locations = nodes[arcs]
    nearby = ((road_locations.max(axis=1) >= south - .001) &
              (road_locations.min(axis=1) <= north + .001)).all(axis=1)
    view = folium.Map(tiles=None, prefer_canvas=True, zoom_control=True)
    folium.TileLayer("OpenStreetMap", show=False).add_to(view)
    for name, lines, color, weight in [
            ("Road network", road_locations[nearby], "#a4b2ac", 1),
            ("Development reference", road_locations[route], "#7659bb", 6),
            ("Decoded sections", [np.asarray(section["coordinates"]) for section in prediction["sections"]],
             "#007c52", 3)]:
        layer = folium.FeatureGroup(name=name).add_to(view)
        for coords in lines:
            folium.PolyLine(coords[:, ::-1], color=color, weight=weight).add_to(layer)
    layer = folium.FeatureGroup(name="GPS observations").add_to(view)
    for point in gps:
        folium.CircleMarker(point[::-1], radius=1.5, stroke=False, fill=True,
                            fill_color="#d96e12", fill_opacity=.8).add_to(layer)
    folium.LayerControl(collapsed=False).add_to(view)
    view.fit_bounds([south[::-1].tolist(), north[::-1].tolist()])
    dest = BASE / "results/demo" / f"development_{trip.name}.html"
    dest.parent.mkdir(exist_ok=True)
    view.save(str(dest))
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train","student_test"], default="train")
    parser.add_argument("--trip", default="00000046")
    args = parser.parse_args()
    if not (len(args.trip) == 8 and args.trip.isdecimal()):
        parser.error("Expected an eight-digit trip ID")
    print(build(ROOT / args.split / args.trip))
