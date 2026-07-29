# Frozen model package

These artifacts were produced by the final full run of
`../train_models_once.py`, completed on 2026-07-25. They allow users to run
`../stress_score_cli.py` immediately without downloading the multi-gigabyte
training CSVs or retraining.

The package contains:

- CatBoost artifacts for SAIDI and SAIFI;
- the selected CatBoost Tweedie compliance model;
- the fitted ANOVA plus Ridge drought pipeline;
- held-out state-year prediction histories;
- final evaluation metrics, model metadata and feature schema;
- `training_complete.json`, which confirms that the saved run completed.

The CLI reads `state_model_history.csv` and `project_metadata.json`. The model
objects are included for research inspection and downstream reuse.

Verify the package with the hashes in `SHA256SUMS.txt`.

Future CLI values are damped recursive baseline scenarios derived from recent
held-out model signals. They are not direct model forecasts using unknown
future weather, population, demand or infrastructure.
