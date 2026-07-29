# BEGINNER REPRODUCIBILITY GUIDE

## U.S. Electricity and Water Stress Monitor

Final RISE 2026 models, frozen artifacts, retraining workflow and offline GUI.

**Authors:** Muhammad Zunair Ali Khan, Sheza Imran and Mohammad Shaffay Asif  
**Faculty supervisor:** Dr. Mudassir Shabbir  
**Research mentors:** Danish Javed and Uzayr Husnain

---

## What this guide lets a beginner do

Choose one of three supported routes:

1. Download and open the one-file offline GUI.
2. Use the included frozen `saved_models` package immediately.
3. Download the three frozen model-ready CSVs and retrain all four pipelines.

This guide reproduces the final poster models and metrics. "Retrain from
scratch" means retraining from the frozen model-ready CSVs. Rebuilding those
tables from changing raw government sources requires a separate raw-source
package and exact source snapshots.

## 1. What is being reproduced

| Pipeline | Prediction | Final model |
|---|---|---|
| SAIDI | Annual outage minutes per customer | Two-stage CatBoost |
| SAIFI | Annual interruptions per customer | Log-CatBoost |
| Drought | Drought severity, 0-100 | ANOVA top 20 + Ridge |
| Compliance | Health-based violation burden per 100,000 residents | CatBoost direct Tweedie |

Electricity stress is 50% SAIDI stress plus 50% SAIFI stress. Water stress is
50% drought severity plus 50% compliance stress.

The final training script begins at the model-ready feature CSV stage.

## 2. Scientific scope and data sources

The modeling study uses U.S. data as a test bed for a framework motivated by
electricity unreliability and water insecurity in Pakistan.

| Source | Role |
|---|---|
| EIA Form EIA-861 | Utility SAIDI/SAIFI targets, customers and reporting |
| Other EIA electricity data | Sales, prices, capacity, generation and demand |
| NOAA Storm Events and climate | Storm exposure, precipitation and temperature |
| Eagle-I | Recorded outage episode context |
| U.S. Drought Monitor | Weekly county D0-D4 drought percentages |
| Census and BEA | Population, poverty, income and GDP |
| EPA SDWIS | Public-water-system and violation records |
| WRI Aqueduct | Structural water-risk context |

The final model-ready grains are:

- electricity: one annual utility-state target represented by twelve
  previous-year monthly input rows;
- drought: one county-week;
- compliance: one public-water-system-year or labeled residual row.

Compliance contributions sum to a state health-based violation burden per
100,000 residents.

## 3. Cleaning, joining and leakage rules

1. Read geographic and entity IDs as text so leading zeroes are preserved.
2. Standardize state FIPS to two digits and county FIPS to five digits.
3. Reduce each context table to the intended join key before merging.
4. Use validated one-to-one or many-to-one joins.
5. Confirm that context joins do not multiply target rows.
6. Sort by entity and time before constructing history.
7. Shift before rolling so the current outcome cannot predict itself.
8. Fit imputation, scaling and feature selection on training rows only.
9. Exclude targets, target decompositions and same-period outcomes.
10. Preserve chronological train, validation and backtest boundaries.

The frozen model-ready tables preserve these rules and prevent changing
external downloads from silently changing the reported results.

## 4. Final model-ready files and features

| File | Approximate size | Training record |
|---|---:|---|
| `electricity_features_final_sequence_2013_2024.csv` | 74 MiB | Utility-state annual sequence |
| `water_drought_features_county_week_2010_2024.csv` | 647 MiB | County-week |
| `water_compliance_features_public_water_system_year_2010_2024.csv` | 835 MiB | System-year or residual |

### SAIDI

A CatBoost classifier estimates the probability of a high-event year.
Separate normal- and high-event CatBoost regressors are blended using that
probability.

### SAIFI

A CatBoost regressor learns `log1p(SAIFI)` and converts predictions back with
`expm1`.

### Drought

Training-only median imputation and scaling are followed by ANOVA
`SelectKBest` with the strongest 20 features and Ridge regression.

### Compliance

Direct CatBoost Tweedie and two-part hurdle candidates compete on 2019-2020
validation data. Direct Tweedie wins and is used for the untouched backtest
and deployment artifact.

## 5. Route A: open the GUI

At the repository root, download:

`Infrastructure_Stress_Monitor.html`

Double-click it. No Python, server or installation is required. The GUI works
offline and contains frozen observed values and precomputed baseline scenarios
for all 50 states plus Washington, D.C., from 2025 through 2050.

The browser does not execute the CatBoost or Ridge models. It displays outputs
prepared from the same final frozen histories and scenario method.

## 6. Route B: use included trained models

### Windows PowerShell

```powershell
git clone https://github.com/amessbee/rise26project.git
Set-Location ".\rise26project\Reproducibility"
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python stress_score_cli.py
```

The included `saved_models/` folder lets the CLI run without downloading the
training tables or retraining. Enter a state name or abbreviation and a
four-digit year after 2024.

The CLI loads `state_model_history.csv` and `project_metadata.json`. Frozen
model objects are included for inspection and downstream reuse.

## 7. Route C: retrain all four pipelines

From `Reproducibility/` with the virtual environment active:

```powershell
python csv_downloader.py
python train_models_once.py --force
python stress_score_cli.py
```

`csv_downloader.py` downloads the three exact filenames into `data/` and
verifies that all are present. `train_models_once.py` uses that relative folder
by default, so no author-specific absolute drive paths are required.

Training is CPU- and memory-intensive. The two water tables contain millions
of rows. Use a well-provisioned computer or university server.

To use CSVs stored elsewhere:

```powershell
python train_models_once.py `
  --electricity-file "X:\data\electricity_features_final_sequence_2013_2024.csv" `
  --drought-file "X:\data\water_drought_features_county_week_2010_2024.csv" `
  --compliance-file "X:\data\water_compliance_features_public_water_system_year_2010_2024.csv" `
  --force
```

## 8. Chronological policy and final results

| Stage | Years | Purpose |
|---|---|---|
| Selection training | Through 2018 | Learn candidates and preprocessing |
| Validation | 2019-2020 | Select settings and compliance family |
| Evaluation refit | Through 2020 | Refit before untouched evaluation |
| Untouched backtest | 2021-2024 | Report final state-year performance |
| Deployment refit | All observed years | Save reusable artifacts |

| Target | MAE | RMSE | R² |
|---|---:|---:|---:|
| SAIDI | 240.555 | 466.147 | 0.083 |
| SAIFI | 0.219 | 0.305 | 0.660 |
| Drought | 0.469 | 0.582 | 0.999 |
| Compliance | 1.648 | 3.406 | 0.805 |

R² is not percentage accuracy. The near-one drought R² is a short-horizon,
lag-rich held-out result, not proof of perfect distant-future forecasts.

## 9. How future scenarios are calculated

The CLI does not run the trained models using imaginary future inputs. It:

1. reads held-out model-signal history;
2. starts from the latest observed state value;
3. takes the median change across the five most recent signals;
4. advances one year at a time;
5. halves the trend after every step;
6. applies a 20% pull toward historical conditions.

These are damped recursive baseline scenarios. They do not include known
future rainfall, storms, population, demand, regulation or infrastructure.
Approximate ranges use backtest RMSE and are not formally calibrated
prediction intervals.

## 10. Reproduction checklist

- `python -m pip install -r requirements.txt` installs `gdown`.
- `saved_models/training_complete.json` exists for the frozen route.
- `python stress_score_cli.py` opens without a missing-package error.
- The downloader verifies all three exact CSV filenames.
- Retraining replaces `saved_models/` only when `--force` is supplied.
- State names and abbreviations are accepted.
- Output includes SAIDI, SAIFI, drought, compliance and both stress scores.
- Compliance is consistently labeled per 100,000 residents.

## Common problems

| Problem | Fix |
|---|---|
| `gdown` not found | Activate `.venv` and reinstall `requirements.txt` |
| Google Drive quota or access error | Retry later or obtain the frozen CSVs from the project owner |
| CSV not found | Run `python csv_downloader.py` and keep exact filenames |
| Memory error | Close other applications or use a stronger computer/server |
| Models already trained | Use the frozen route, or add `--force` to retrain |
| CLI metadata missing | Restore the committed `saved_models/` folder |

## Final interpretation

A successful reproduction means either:

- the included frozen package returns electricity and water scenarios; or
- the downloaded frozen CSVs recreate the four pipelines and their saved
  artifacts, after which the CLI returns both scores.

The system is for research comparison and scenario exploration. It is not an
official utility, public-health or emergency-management forecast.
