# Run the PathSnap evaluator

Run commands from the cloned repository. Laptop 1 needs the numbered trip directories and the checked-in model bundle. Laptop 2 needs only Python 3, the ui folder and ui_server.py. No cloud services or tile downloads are required by the evaluator UI. Run only on a trusted private LAN; the service is not authenticated.

## Laptop 1: inference and scoring

PowerShell, using Python 3.13 (or an existing compatible virtual environment):

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-demo.txt
$env:PATHSNAP_DATA = (Resolve-Path '..\Student folder\Student folder').Path
$env:PATHSNAP_TRIPS = Join-Path $env:PATHSNAP_DATA student_test
.\.venv\Scripts\python -m uvicorn api:app --host 0.0.0.0 --port 8000 --access-log
```

Adjust the data path if your folders are elsewhere. For the physical evaluation, set PATHSNAP_TRIPS to the reserved split and restart. Do not train or tune on it. Leave the receiving terminal visible to show live requests. The predictor reads only road and GPS inputs; scoring runs afterwards on Laptop 1. A missing reference returns a route with “Not scored,” not a made-up score.

## Laptop 2: evaluator interface

On the same private Wi-Fi, replace the example IP with Laptop 1's IPv4 address from ipconfig. Allow inbound TCP 8000 on Laptop 1's private network only if needed. Guest-network client isolation can prevent the connection.

```powershell
python ui_server.py --api http://192.168.1.20:8000
```

Open http://127.0.0.1:8501 on Laptop 2. Select any trip and reconstruct. Zoom, drag, fit the route or toggle GPS and decoded roads. Separate decoded sections are never joined across breaks. The UI forwards requests to Laptop 1; it does not read student files, reference geometry or stored predictions. For a single-machine check, use 127.0.0.1 instead of the example LAN IP.

## Verify and generate development evidence

```powershell
.\.venv\Scripts\python -m unittest test_decoder test_demo
.\.venv\Scripts\python rehearse.py --api http://127.0.0.1:8501
.\.venv\Scripts\python -m pip install folium==1.0.0 playwright
.\.venv\Scripts\python development_map.py
```

The rehearsal checks all ten student-test responses against the stored refinement scores. The Folium HTML in results/demo overlays local roads, GPS, development reference and decoded sections for training trip 00000046. It uses web-hosted Leaflet assets; internet access is required to open the Folium export. Reference-inclusive maps are development outputs only and are not served by FastAPI.

To rebuild the PDF, run both services, generate the rehearsal and development map first, and capture the API process's real logs (e.g. append `2> api-run.log` to the uvicorn command). With Edge installed run:

```powershell
.\.venv\Scripts\python report.py --log api-run.log
```

The report and screenshots are written to results/demo. These results concern student data only. Review the report, include team details and AI disclosure as required, then add genuine screenshots from the physical two-laptop demo before final submission. For full model retraining use the original requirements and experiment sequence in README.md on Linux/WSL (the experiment writers use POSIX file locks).
