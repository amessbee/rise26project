# Reproducibility

This folder supports two reproducibility routes for the final RISE project.

## Route A: use the frozen results

The included `saved_models/` package contains the final trained artifacts,
metadata, evaluation metrics and held-out state-year history.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python stress_score_cli.py
```

No training data download is required for this route.

## Route B: retrain all four pipelines

The three model-ready CSVs are hosted outside GitHub because two exceed
GitHub's normal file-size limit.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python csv_downloader.py
python train_models_once.py --force
python stress_score_cli.py
```

Training is CPU- and memory-intensive. The compliance and drought tables
contain millions of rows.

## GUI

The repository root contains `Infrastructure_Stress_Monitor.html`. Download
that one file and double-click it to explore the frozen projections without
Python, a server or model retraining.

## Documentation

- `Beginner_Reproducibility_Guide.pdf`: designed step-by-step guide.
- `Beginner_Reproducibility_Guide.md`: editable source with the same content.

The guide defines reproduction from the frozen model-ready datasets. A full
raw-government-source rebuild additionally requires exact source snapshots and
separate data-engineering scripts, which are not bundled here.
