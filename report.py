"""Build the development report from the supplied data and recorded experiment outputs."""
import argparse
import csv
import json
import os
from html import escape
from pathlib import Path
from statistics import mean

import numpy as np
from playwright.sync_api import sync_playwright

from roads import ROOT, load_trip

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
OUT = RESULTS / "demo"


def read(name):
    return json.loads((RESULTS / name).read_text())


def table(headers, rows):
    def cells(tag, values):
        return "".join(f"<{tag}>{escape(str(value))}</{tag}>" for value in values)
    return "<table><tr>" + cells("th", headers) + "</tr>" + "".join(
        "<tr>" + cells("td", row) + "</tr>" for row in rows) + "</table>"


def bars(pairs):
    return "".join(f"<div class='bar'><span>{escape(name)}</span><i style='width:{score*48}%'></i> {score:.1%}</div>"
                   for name, score in pairs)


def page(kicker, title, body):
    return f"<section><small>PATHSNAP &nbsp; / &nbsp; {kicker}</small><h1>{title}</h1>{body}</section>"


def build(ui_url, log):
    OUT.mkdir(exist_ok=True)
    rows = list(csv.DictReader((RESULTS / "comparison.csv").open()))
    tests = [row for row in rows if row["split"] == "student_test"]
    final = read("refinement/comparison.json")
    tuning = read("tuning/comparison.json")
    study = read("tuning/study.json")
    decoder = read("decoder/comparison.json")["student_test"]
    replay = read("demo/rehearsal.json")
    sections = []
    inventory = []
    train_sizes = []
    for split in ("train", "student_test"):
        sizes, gaps, bad, invalid = [], [], 0, 0
        for folder in sorted((ROOT / split).glob("[0-9]*")):
            nodes, arcs, track = load_trip(folder)
            reference = np.loadtxt(folder / f"{folder.name}.route", dtype=int, ndmin=1)
            sizes.append(len(track))
            gaps.extend(np.diff(track[:, 2]).tolist())
            bad += int((~np.isfinite(track)).sum() + (~np.isfinite(nodes)).sum())
            invalid += int(np.count_nonzero((reference < 0) | (reference >= len(arcs))))
        inventory.append([split,len(sizes),sum(sizes),min(sizes),max(sizes),bad,invalid,
                          sum(value > 30 for value in gaps)])
        if split == "train":
            train_sizes = sizes
    sections.append(page("01 / DATA", "From GPS to directed roads",
        "<p class='lead'>Recover the road taken by a vehicle, not merely the closest line on a map.</p>"+
        table(["Split","Trips","Readings","Min/trip","Max/trip","Nonfinite GPS/node values","Invalid route IDs","Gaps >30 s"],inventory)+
        "<h2>Trip length distribution · training only</h2>"+
        bars([(f"{low:,}–{high-1:,} readings",sum(low<=count<high for count in train_sizes)/len(train_sizes))
              for low, high in [(0,1000),(1000,2500),(2500,5000),(5000,10000),(10000,20000)]])+
        "<h2>Labels are inferred, not measured</h2><p>Node rows identify positions and arc rows specify directed node pairs. A reference route gives an ordered sequence of arc IDs, not one label per timestamp. Dynamic programming aligns observations to nondecreasing route indices by minimizing total point-to-segment distance. Repeated geometry and stops introduce ambiguity. Accuracy below means agreement with that alignment.</p>"
        "<h2>Preventing leakage</h2><p>All candidate searches and feature extraction use only GPS and road geometry. Latitude and longitude are projected into a local metre-based plane. Whole trips stay together in three training folds. Ten student-test trips support development comparisons; no reserved evaluator trip contributes to this report.</p>"))
    model_rows = []
    for row in tests:
        name = row["model"]
        cv = mean(float(item["macro_f1"]) for item in rows if item["model"] == name and item["split"].startswith("cv"))
        model_rows.append([name,f"{cv:.4f}",f"{float(row['top1']):.2%}",f"{float(row['macro_f1']):.4f}",
                           row["binary_log_loss"][:6] or "N/A*",f"{float(row['inference_s']):.4f}"])
    sections.append(page("02 / MODEL COMPARISON", "Which road fits this observation?",
        "<p>Each GPS reading has five candidates in the initial comparison. Top-1 counts a missing reference candidate as wrong. Macro F1 averages positive and negative candidate classes after exactly one winner is selected. Log loss assesses candidate probabilities; timings cover scoring only.</p>"+
        table(["Algorithm","CV F1","Test Top-1","Test F1","Log loss","Seconds"],model_rows)+
        "<p>*Nearest road is a ranking rule without a fitted probability. Initial per-model road continuity and training curves were not recorded; continuity is measured in the decoder experiment. Fold differences are not evidence of overfitting without comparable training scores.</p>"
        "<h2>Model decisions</h2><p><b>Logistic regression:</b> standardized geometry, logistic loss, up to 250 iterations. Fast and interpretable but treats road features additively. <b>Decision tree:</b> Gini splits, depth 10, minimum 32 per leaf. Captures thresholds but gives the weakest test Top-1 of the trained initial models.</p>"
        "<p><b>Random forest:</b> 36 trees, depth 16, minimum 10 per leaf. Averaging helps versus one tree, at a higher scoring cost. <b>Histogram boosting:</b> 100 iterations of binary log loss, no early stopping. Nonlinear interactions become more useful with richer geometry.</p>"
        "<p><b>Directed GraphSAGE:</b> one-hop mean road features, 15 epochs, binary cross entropy, AdamW at 0.003. The matched no-neighbour ablation slightly exceeds its test Top-1; these results do not show that adding graph neighbours improves candidate ranking.</p>"))
    enhanced = read("improvement_selection.json")
    fold_rows = []
    for row in tests:
        folds = [item for item in rows if item["model"] == row["model"] and item["split"].startswith("cv")]
        fold_rows.append([row["model"], *[f"{float(item['top1']):.2%}" for item in folds],
                          f"{float(row['candidate_miss']):.2%}"])
    sections.append(page("DIAGNOSTICS / CROSS-VALIDATION","Fold variation and probability quality",
        table(["Model","Fold 1 Top-1","Fold 2 Top-1","Fold 3 Top-1","Student candidate miss"],fold_rows)+
        "<p>The same five candidates enter every initial model. A reference absent from the set cannot be recovered by changing the classifier. Fold 1 has substantially more candidate misses, so its weaker Top-1 scores should not automatically be attributed to model overfitting. No held-out evaluator labels were consulted.</p>"
        "<h2>Student-test binary log loss · lower is better</h2>"+
        "".join(f"<div class='bar'><span>{escape(row['model'])}</span><i style='width:{float(row['binary_log_loss'])*150:.2f}%'></i> {float(row['binary_log_loss']):.4f}</div>"
                for row in tests if row["binary_log_loss"])+
        "<p>These scores compare probability quality over individual candidates. They are not sequence probabilities. The nearest-distance heuristic has no fitted probabilities, and the decoder selects a sequence rather than outputting calibrated posteriors.</p>"))
    sections.append(page("03 / FEATURE AND MODEL SELECTION","Geometry matters more than model size",
        "<p>The first comparison sampled up to 60,000 complete observations per fit (seed 24). An expanded experiment used all training observations. Along-road position, distances to endpoints, relative distance and before/after GPS context at four window sizes supplement distance, heading, speed, length, orientation and endpoint degrees. Centred windows require a completed trip.</p>"
        "<h2>Three-fold candidate Macro F1 · all training observations</h2>"+
        bars(list(enhanced["cv_macro_f1"].items()))+
        "<p>The context MLP ranks candidates jointly; Context GraphSAGE adds neighbour summaries. Both use candidate cross-entropy for training observations that contain the reference. The validation denominator still includes misses. Geometry boosting wins the recorded selection; held-out comparisons for the unselected enhanced variants were not recorded.</p>"
        "<h2>Student-test Top-1 · original comparison</h2>"+
        bars([(row["model"], float(row["top1"])) for row in tests])))
    winner=tuning["comparison"][-1]
    search=study["specification"]["search_space"]
    sections.append(page("04 / OPTIMISATION","Bayesian search and refit",
        "<p>A Gaussian process with a Matérn kernel proposes settings by expected improvement. The objective is mean three-fold candidate Macro F1, with whole trips kept together. The baseline and five random initialisations precede 18 Bayesian proposals. The study and refit took %.1f minutes.</p>"%(tuning["study_and_refit_seconds"]/60)+
        table(["Parameter","Search space","Selected"],[[key,str(spec.get("values") or [spec.get("low"),spec.get("high")]),str(winner["parameters"][key])[:12]] for key,spec in search.items()])+
        table(["Model","CV F1","Test Top-1","Test F1","Test loss","Scoring s"],[[row["model"],f"{row['cv_macro_f1']:.4f}",f"{row['student_test']['top1']:.2%}",f"{row['student_test']['macro_f1']:.4f}",f"{row['student_test']['binary_log_loss']:.4f}",f"{row['inference_seconds']:.3f}"] for row in tuning["comparison"]])+
        "<p>The selected configuration raises student-test accuracy by about 0.15 percentage points, not a large step change. Trial-by-trial settings, fold scores and search metadata are retained in results/tuning/study.json.</p>"))
    sections.append(page("05 / ROUTE RECONSTRUCTION","From independent scores to a route",
        "<p>Hidden state: a directed arc and the GPS projection along it. Emissions normalize classifier probabilities across available candidates. Transitions compare directed shortest-path distance with observed GPS displacement. Viterbi accumulates log scores and backtracks the best sequence; connector arcs fill reachable gaps between roads.</p>"
        "<p>No candidate yields an unmatched reading. Unreachable transitions, non-increasing time and gaps over 30 seconds split the route. Sections remain separate on the map. A bounded speed rule limits path search; 20 metres of backward projection jitter is tolerated on the same arc.</p>"+
        table(["Pipeline","Top-1","Continuity","Mean sequence similarity"],[[label,f"{row['top1']:.2%}",f"{row['route_continuity']:.2%}",f"{row['mean_arc_sequence_similarity']:.4f}"] for label,row in [("Five / classifier",decoder["classifier"]),("Five / decoded",decoder["decoder"]),("Twelve / decoded",final["refined"])]])+
        "<p>Continuity counts reachable adjacent predictions within the routing bound. Sequence similarity is one minus normalized edit distance after collapsing adjacent duplicate arc IDs. At width five, trip 00000091 lost one correct reading after decoding despite improved continuity. A smoother route is not necessarily a more accurate point-level match.</p>"
        "<h2>Final configuration</h2><p>Expanding to twelve candidates improves candidate coverage to 99.88%. Three grouped folds selected the existing 31 features, beta of 5 metres and transition strength 0.3. Six of 19,148 eligible student-test transitions remain disconnected. Model selection and decoder evaluation are distinct stages.</p>"))
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=os.environ.get("PATHSNAP_BROWSER","msedge"))
        tab = browser.new_page(viewport={"width":1280,"height":850}, device_scale_factor=1)
        tab.goto(ui_url)
        tab.locator("#trip").select_option("00000077")
        tab.locator("#run").click()
        tab.wait_for_function("document.getElementById('status').textContent.includes('complete')",timeout=180000)
        tab.screenshot(path=str(OUT/"workspace.png"))
        tab.goto((OUT/"development_00000046.html").as_uri(),wait_until="domcontentloaded")
        tab.wait_for_selector(".leaflet-overlay-pane canvas, .leaflet-overlay-pane svg",timeout=90000)
        tab.wait_for_selector(".leaflet-control-layers")
        tab.wait_for_timeout(700)
        tab.screenshot(path=str(OUT/"folium.png"))
        sections.append(page("06 / GEOGRAPHIC INSPECTION","A training trip in four layers",
            "<p>Trip 00000046 · local road geometry (grey), reference (violet), reconstructed route (green), and raw GPS (orange). Reference geometry is present only in this offline development map. The interactive HTML includes independent layer controls.</p><img src='folium.png'><p>At full-trip scale the layers often overlap. Zoom and toggle the reference and prediction independently to inspect local disagreements.</p>"))
        lines = [line.strip() for line in log.read_text(errors="replace").splitlines() if "trip=" in line or "/api/predict/" in line]
        if not lines:
            raise ValueError("A real receiving-server log is required")
        sections.append(page("07 / DEPLOYMENT","Evaluator workspace and receiving API",
            "<p>Actual local rehearsal using separate UI and API processes. The UI forwards a selected trip ID; FastAPI loads the saved classifier, predicts from GPS and road data, and only then reads the reference to return aggregate agreement. The live UI does not receive the reference geometry.</p><img src='workspace.png'><h2>Receiving process · actual log excerpt</h2><pre>"+escape("\n".join(lines[-4:]))+"</pre><p>Same-machine integration evidence, not a photograph of a two-laptop evaluation. Capture the physical receiving terminal and UI at the final LAN demo.</p>"))
        sections.append(page("08 / VERIFICATION","Rehearsal across the development split",
            table(["Student trip","GPS readings","Agreement","Sections","Inference s"],[[row["trip"],row["observations"],f"{row['accuracy']:.2%}",row["sections"],f"{row['inference_seconds']:.3f}"] for row in replay])+
            "<p>Every response was computed by the live API, not read from the prediction archive. The rehearsal checked prediction counts and agreement against the stored development experiment for each trip. Timing excludes server-side reference alignment, transport and rendering.</p>"
            "<h2>Reproduction and limits</h2><p>See README.md for data configuration, training commands and two-laptop launch steps. The report reads the original metrics from results/, inventories the local dataset and captures the real UI and Folium map. No reserved evaluator results are reported. Before submission, the team should check the methodology and attribution, add its identifying details and capture the actual two-laptop demonstration.</p>"))
        css = """@page{size:A4;margin:16mm}*{box-sizing:border-box}body{margin:0;color:#18332e;font:10.5pt/1.48 Arial,sans-serif}section{break-after:page}section:last-child{break-after:auto}small{font:bold 9pt monospace;color:#32735c}h1{font-size:26pt;letter-spacing:-1pt;line-height:1.1;margin:10pt 0 16pt}h2{font-size:14pt;margin:16pt 0 5pt}.lead{font-size:14pt}table{width:100%;border-collapse:collapse;font-size:9pt;margin:10pt 0}td,th{text-align:left;padding:6pt;border-bottom:1px solid #cbdad2}th{background:#e3f1e9}img{width:100%;max-height:127mm;object-fit:contain;border:1px solid #d3ded7}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#14332c;color:white;padding:9pt;font-size:8pt}.bar{margin:5pt 0;font:9pt monospace}.bar span{display:inline-block;width:35%}.bar i{display:inline-block;height:7pt;background:#328564}"""
        html = OUT/"report.html"
        html.write_text("<meta charset=utf-8><title>PathSnap technical report</title><style>"+css+"</style>"+"".join(sections),encoding="utf-8")
        tab.goto(html.as_uri())
        tab.pdf(path=str(OUT/"PathSnap-report.pdf"),prefer_css_page_size=True,print_background=True)
        browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ui",default="http://127.0.0.1:8501")
    parser.add_argument("--log",type=Path,required=True)
    args = parser.parse_args()
    build(args.ui,args.log)
