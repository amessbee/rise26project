# Project Overview

This project trains, validates, and freezes four machine learning models that estimate infrastructure and environmental stress across U.S. states over time:

- **Electrical grid reliability** â€” SAIDI (outage duration) and SAIFI (outage frequency) per customer
- **Drought severity** â€” a 0-100 severity score at the county/state level
- **Drinking water compliance** â€” health-based violation rates for public water systems

The models are built from three separate feature datasets (electricity, drought, and water compliance - sourced and described in the Reproducibility section) and produce state-year level predictions that can be used to track historical stress trends and generate short-term baseline projections. All four pipelines are trained, evaluated, and saved in a single run, producing a self-contained `saved_models/` folder with the trained artifacts, performance metrics, and prediction history that downstream tools (like the CLI scorer and GUI) read from.

# What's in the Repository

- **`train_models_once.py`** - the main training script. Runs all four pipelines end to end, evaluates them chronologically, and saves the final models, metrics, and prediction histories into `saved_models/`.
- **`stress_score_cli.py`** - a command-line tool that loads the saved models and produces stress scores/projections without retraining anything.
- **A GUI file** - a graphical interface for exploring the stress scores/projections without using the command line.
- **`saved_models/`** - the output folder created by `train_models_once.py`. Contains the trained model artifacts (`.cbm`, `.joblib`), prediction history CSVs, evaluation metrics, feature schemas, and a `training_complete.json` marker confirming a full run finished successfully.
- **`reproducibility/`** - everything needed for someone else to reproduce these results from scratch: a step-by-step walkthrough of the full process (downloading the raw data, generating the feature CSVs, training the models, and reproducing the reported metrics), a script that downloads the source CSVs, and the project's dependency requirements.

# Models & Methodology

**SAIDI** is predicted with a two-stage CatBoost approach: a classifier first flags high-severity outage years, then separate CatBoost regressors are trained on the normal and high subsets. Final predictions blend the two regressors' outputs, weighted by the classifier's probability.

**SAIFI** uses a single log-CatBoost regressor - a CatBoost model trained on log-transformed outage-frequency targets, with predictions exponentiated back to the original scale.

**Drought** severity is modeled with ANOVA SelectKBest feature selection followed by Ridge regression. Candidate features are ranked by their univariate correlation with the target using an ANOVA F-test, the strongest ones are kept, and a Ridge regression is fit on the standardized, imputed feature set.

**Compliance** violation rates are modeled by letting two candidate approaches compete on the validation split and keeping whichever performs better: a single CatBoost regressor trained with a Tweedie loss, suited to zero-inflated, right-skewed rate data, versus a two-part hurdle model that pairs a classifier predicting whether a violation occurs with a regressor predicting its magnitude if so. The winning family is the one actually deployed.

Across all four, only leakage-safe features are used. Columns that wouldn't be known at prediction time, such as the targets themselves or same-period outcome variables, are explicitly excluded during feature preparation, and this exclusion is checked by an automated audit before training.

# Chronological Evaluation Policy

Models are trained and scored using chronological splits rather than random ones, so reported performance reflects how they'd actually perform on real future data - earlier years are used for training/selection, and only later, untouched years are used to measure final accuracy. Because of this, the historical "predicted" columns in the output CSVs are genuinely out-of-sample, and years too early to have an out-of-sample prediction are intentionally left null rather than filled with in-sample fits.

## Open the GUI

### Use online â€” no installation

[**Open the live U.S. Infrastructure Stress Monitor â†’**](https://amessbee.github.io/rise26project/)

The browser version uses the frozen, precomputed project outputs included in
the repository.

### Run locally on Windows

Download or clone the repository, then double-click:

```text
START_GUI_WINDOWS.bat
```

Open `http://127.0.0.1:8081` if the browser does not open automatically.

### Run locally on macOS

Download or clone the repository. In Terminal, from the repository folder, run:

```bash
chmod +x START_GUI_MAC.command
./START_GUI_MAC.command
```

Open `http://127.0.0.1:8081` if the browser does not open automatically.

