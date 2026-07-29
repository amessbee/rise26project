# Forecasting Electricity and Water Stress Across U.S. States Using Machine Learning

Final research project for the **LUMS Research Internship in Science and
Engineering (RISE), Computer Science, Summer 2026**.

**Authors:** Muhammad Zunair Ali Khan, Sheza Imran and Mohammad Shaffay Asif  
**Faculty supervisor:** Dr. Mudassir Shabbir  
**Research mentors:** Danish Javed and Uzayr Husnain

## Project overview

This project combines public electricity reliability, climate, drought,
population and drinking-water compliance records into two comparable
state-level indicators:

- **Electricity stress:** 50% outage-duration stress (SAIDI) and 50%
  outage-frequency stress (SAIFI).
- **Water stress:** 50% drought severity and 50% drinking-water compliance
  stress.

The project began from the practical problem of electricity unreliability and
water insecurity in Pakistan. Because consistent machine-readable local
history was limited, the modeling study uses granular U.S. public records as a
test bed for the framework.

## Open the GUI

[**Download the one-file U.S. Infrastructure Stress Monitor**](https://raw.githubusercontent.com/amessbee/rise26project/refs/heads/main/Infrastructure_Stress_Monitor.html)

Download `Infrastructure_Stress_Monitor.html` and double-click it. The GUI:

- works offline in a normal browser;
- requires no Python installation or local server;
- includes frozen observed and baseline-scenario values;
- supports all 50 states plus Washington, D.C., for 2025-2050.

The GUI uses precomputed outputs from the final frozen run. It does not execute
CatBoost or Ridge models inside the browser.

## Final models and held-out results

Model selection used 2019-2020 validation data. Final evaluation used an
untouched 2021-2024 state-year backtest after refitting through 2020.

| Target | Final model | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| SAIDI | Two-stage CatBoost high/normal probability-weighted blend | 240.555 | 466.147 | 0.083 |
| SAIFI | Log-CatBoost regressor | 0.219 | 0.305 | 0.660 |
| Drought | ANOVA SelectKBest (top 20) plus Ridge regression | 0.469 | 0.582 | 0.999 |
| Compliance | CatBoost direct Tweedie | 1.648 | 3.406 | 0.805 |

Compliance is measured as the state health-based violation burden per
**100,000 residents**.

R² is not percentage accuracy. Drought's high held-out R² reflects a
short-horizon, lag-rich state-year backtest and should not be interpreted as
proof of nearly perfect distant-future forecasts.

## Chronological evaluation policy

| Stage | Years | Purpose |
|---|---|---|
| Selection training | Through 2018 | Learn candidate models and preprocessing |
| Validation | 2019-2020 | Choose settings and the compliance model family |
| Evaluation refit | Through 2020 | Refit without using backtest outcomes |
| Untouched backtest | 2021-2024 | Report final held-out performance |
| Deployment refit | All observed years | Save artifacts for reuse |

Historical predicted columns are genuinely out of sample: 2019-2020 signals
come from models trained through 2018, and 2021-2024 signals come from frozen
evaluation models refitted through 2020. Earlier predicted values are left
null rather than replaced with in-sample fits.

## Two reproducibility routes

### A. Use the included trained models

The repository includes `Reproducibility/saved_models/`. This is the quickest
route and does not require the multi-gigabyte training CSVs:

```powershell
git clone https://github.com/amessbee/rise26project.git
Set-Location ".\rise26project\Reproducibility"
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python stress_score_cli.py
```

### B. Retrain all four pipelines

The downloader retrieves the three frozen model-ready CSVs from the project's
public Google Drive folder into `Reproducibility/data/`:

```powershell
Set-Location ".\rise26project\Reproducibility"
.\.venv\Scripts\Activate.ps1
python csv_downloader.py
python train_models_once.py --force
python stress_score_cli.py
```

Training is resource-intensive because the drought and compliance tables
contain millions of records.

## What the two Python programs do

### `train_models_once.py`

- validates the three model-ready tables and audits leakage;
- trains the four final pipelines with chronological splits;
- aggregates utility, county and water-system predictions to state-year;
- saves model objects, held-out histories, metrics, feature schemas and
  metadata into `saved_models/`;
- records package versions, input fingerprints and the final-run status.

### `stress_score_cli.py`

The CLI does **not** run the CatBoost and Ridge models against unknown future
features. It reads the frozen out-of-sample state-year model-signal history,
measures each state's recent direction, predicts one year at a time, halves
the trend after each step and pulls values gradually toward historical
conditions.

These are **damped recursive baseline scenarios**, not forecasts with known
future weather, population, demand, regulation or infrastructure. Approximate
ranges use held-out RMSE and are not formally calibrated prediction intervals.

## Repository structure

```text
rise26project/
|-- Infrastructure_Stress_Monitor.html
|-- README.md
`-- Reproducibility/
    |-- README.md
    |-- Beginner_Reproducibility_Guide.pdf
    |-- Beginner_Reproducibility_Guide.md
    |-- requirements.txt
    |-- csv_downloader.py
    |-- train_models_once.py
    |-- stress_score_cli.py
    `-- saved_models/
        |-- training_complete.json
        |-- evaluation_metrics.json
        |-- project_metadata.json
        |-- feature_schema.json
        |-- state_model_history.csv
        `-- model artifacts and supporting histories
```

## Data sources

The model-ready tables were assembled from public records including EIA-861,
EIA electricity data, NOAA/NCEI climate and Storm Events, Eagle-I outage
records, the U.S. Drought Monitor, U.S. Census and BEA socioeconomic data,
EPA SDWIS drinking-water records and WRI Aqueduct context.

The reproducibility download contains frozen model-ready tables so results do
not change when external government sources are revised. Rebuilding every
table from raw sources additionally requires the exact source snapshots and
separate data-engineering code; that broader raw-source package is outside
this repository.

## Detailed guide

- [Read the designed PDF guide](./Reproducibility/Beginner_Reproducibility_Guide.pdf)
- [Read or edit the Markdown guide](./Reproducibility/Beginner_Reproducibility_Guide.md)

## Research-use warning

The scores support comparison and scenario exploration. They are not official
utility, public-health or emergency-management forecasts and should not be
used as the sole basis for operational decisions.
