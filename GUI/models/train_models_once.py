"""
Train, evaluate, freeze, and save the four U.S. stress-project models.

Chronological policy
--------------------
Selection training: through 2018
Validation:         2019-2020
Evaluation refit:   through 2020
Untouched backtest: 2021-2024
Deployment refit:   all observed years

The historical predicted columns written for the CLI are out-of-sample signals:
2019-2020 come from the selection models trained through 2018, and 2021-2024
come from the frozen evaluation models trained through 2020. Earlier years are
left null rather than filled with in-sample fitted values.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import shutil
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

try:
    import catboost
except ImportError as error:  # pragma: no cover - import already required above
    raise ImportError(
        "CatBoost is required. Install dependencies with: "
        "py -3.13 -m pip install pandas numpy scikit-learn catboost joblib"
    ) from error


# -----------------------------------------------------------------------------
# Fixed project paths and chronological periods
# -----------------------------------------------------------------------------

ELECTRICITY_FILE = (
    r"D:\RISE_Project\Feature_CSVs"
    r"\electricity_features_final_sequence_2013_2024.csv"
)

DROUGHT_FILE = (
    r"D:\RISE_Project\Feature_CSVs"
    r"\water_drought_features_county_week_2010_2024.csv"
)

COMPLIANCE_FILE = (
    r"D:\RISE_Project\Feature_CSVs"
    r"\water_compliance_features_public_water_system_year_2010_2024.csv"
)

PROJECT_FOLDER = Path(__file__).resolve().parent
MODEL_FOLDER = PROJECT_FOLDER / "saved_models"

RANDOM_SEED = 67
TRAIN_END_YEAR = 2018
VALIDATION_YEARS = (2019, 2020)
VALIDATION_END_YEAR = 2020
BACKTEST_YEARS = (2021, 2022, 2023, 2024)

SAIDI = "target_saidi_minutes_per_customer"
SAIFI = "target_saifi_interruptions_per_customer"
CUSTOMERS = "sample_weight_reporting_customers"

DROUGHT = "target_drought_severity_0_100"
DROUGHT_WEIGHT = "county_land_area_weight_within_state"

COMPLIANCE = (
    "target_health_based_violation_rate_contribution"
    "_per_100000_state_residents"
)
COMPLIANCE_COUNT = "target_health_based_violation_count"
COMPLIANCE_POPULATION = "state_population_persons_for_target"
COMPLIANCE_RESIDUAL_FLAG = "unallocated_state_year_residual_flag"

ELECTRICITY_MONTHLY_COLUMNS = [
    "eia861m_total_sales_mwh",
    "eia861m_total_customers",
    "service_territory_county_count",
    "nrel_allocated_monthly_demand_mwh",
    "noaa_allocated_storm_event_count",
    "noaa_allocated_storm_duration_hours",
    "noaa_allocated_storm_property_damage_usd",
    "eaglei_allocated_outage_episode_count",
    "eaglei_allocated_outage_duration_hours",
    "eaglei_allocated_customer_hours_out",
    "eaglei_allocated_max_customers_out",
    "service_territory_precipitation_inches",
    "service_territory_temperature_f_mean",
    "service_territory_cooling_degree_days",
    "service_territory_monthly_load_factor",
    "service_territory_mapping_available_flag",
    "noaa_service_territory_climate_available_flag",
    "nrel_county_demand_available_flag",
    "eaglei_source_year_available_flag",
]

ELECTRICITY_CATEGORICAL_COLUMNS = [
    "utility_number",
    "state_fips",
    "ownership",
    "reporting_standard",
]

COMPLIANCE_PREFERRED_CATEGORICAL_COLUMNS = [
    "state_abbreviation",
    "master_record_type",
]

HISTORY_MARKERS = (
    "previous",
    "prior",
    "lag",
    "rolling",
    "history",
    "historical",
    "recent",
    "consecutive",
    "repeat",
    "change",
    "growth",
    "mean",
    "median",
    "max",
    "min",
    "std",
    "trend",
    "years_since",
)

MANDATORY_COMPLIANCE_HISTORY_FEATURES = [
    "previous_year_had_health_based_violation_flag",
    "years_since_previous_health_based_violation",
    "previous_3_year_total_health_based_violation_count",
]

MODEL_LABELS = {
    "saidi": "Two-stage CatBoost high/normal probability-weighted blend",
    "saifi": "Log-CatBoost regressor",
    "drought": "ANOVA SelectKBest plus Ridge regression",
}


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TweedieConfiguration:
    variance_power: float
    depth: int
    learning_rate: float

    @property
    def name(self) -> str:
        power = str(self.variance_power).replace(".", "_")
        rate = str(self.learning_rate).replace(".", "_")
        return f"tweedie_p{power}_d{self.depth}_lr{rate}"


@dataclass(frozen=True)
class HurdleConfiguration:
    classifier_depth: int = 6
    classifier_learning_rate: float = 0.05
    magnitude_depth: int = 6
    magnitude_learning_rate: float = 0.05

    @property
    def name(self) -> str:
        classifier_rate = str(self.classifier_learning_rate).replace(".", "_")
        magnitude_rate = str(self.magnitude_learning_rate).replace(".", "_")
        return (
            f"hurdle_classifier_d{self.classifier_depth}_lr{classifier_rate}"
            f"_magnitude_d{self.magnitude_depth}_lr{magnitude_rate}"
        )


@dataclass
class ComplianceCandidateResult:
    candidate: str
    family: str
    configuration: dict[str, Any]
    iterations: dict[str, int]
    validation_state_metrics: dict[str, Any]
    validation_system_metrics: dict[str, Any]
    classifier_diagnostics: dict[str, Any] | None
    training_seconds: float
    prediction_seconds: float
    state_predictions: pd.DataFrame


# -----------------------------------------------------------------------------
# Argument parsing and atomic output management
# -----------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train, chronologically evaluate, and freeze the four stress models."
        )
    )
    parser.add_argument(
        "--electricity-file",
        default=ELECTRICITY_FILE,
        help="Electricity feature CSV path.",
    )
    parser.add_argument(
        "--drought-file",
        default=DROUGHT_FILE,
        help="Drought feature CSV path.",
    )
    parser.add_argument(
        "--compliance-file",
        default=COMPLIANCE_FILE,
        help="Compliance feature CSV path.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=6,
        help="CatBoost CPU thread count. Default: 6.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a reduced non-final schema and pipeline smoke test.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing saved_models folder after a successful run.",
    )
    parser.add_argument(
        "--compliance-tuning-rows",
        type=int,
        default=250_000,
        help="Maximum compliance training rows used during validation tuning.",
    )
    parser.add_argument(
        "--compliance-eval-rows",
        type=int,
        default=100_000,
        help="Maximum validation rows used for CatBoost early stopping.",
    )
    parser.add_argument(
        "--max-compliance-numeric-features",
        type=int,
        default=80,
        help="Maximum numeric compliance features retained after leakage audit.",
    )
    parser.add_argument(
        "--prediction-chunk-size",
        type=int,
        default=200_000,
        help="Rows predicted at a time for large compliance data.",
    )
    return parser.parse_args()


def configure_threads(threads: int) -> None:
    if threads < 1:
        raise ValueError("--threads must be at least 1.")
    value = str(threads)
    os.environ.setdefault("OMP_NUM_THREADS", value)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", value)
    os.environ.setdefault("MKL_NUM_THREADS", value)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", value)


def prepare_output_folder(force: bool) -> Path:
    complete_file = MODEL_FOLDER / "training_complete.json"

    if complete_file.exists() and not force:
        print("The four pipelines have already been trained.")
        print("Saved models:", MODEL_FOLDER)
        print()
        print("To use them, run:")
        print("py -3.13 stress_score_cli.py")
        print()
        print("To deliberately train everything again, run:")
        print("py -3.13 train_models_once.py --force")
        raise SystemExit(0)

    if MODEL_FOLDER.exists() and not force:
        raise FileExistsError(
            f"An incomplete or unverified output folder already exists:\n{MODEL_FOLDER}\n"
            "Re-run with --force only when you intend to replace it."
        )

    temporary = PROJECT_FOLDER / f".saved_models_tmp_{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=False)
    return temporary


def publish_output_folder(temporary: Path, force: bool) -> None:
    backup = PROJECT_FOLDER / f".saved_models_backup_{os.getpid()}"

    if backup.exists():
        shutil.rmtree(backup)

    moved_old = False
    try:
        if MODEL_FOLDER.exists():
            if not force:
                raise FileExistsError(
                    "The saved_models folder appeared during training. "
                    "It was not replaced because --force was not supplied."
                )
            os.replace(MODEL_FOLDER, backup)
            moved_old = True

        os.replace(temporary, MODEL_FOLDER)

        if moved_old and backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if not MODEL_FOLDER.exists() and moved_old and backup.exists():
            os.replace(backup, MODEL_FOLDER)
        raise


def cleanup_temporary_folder(temporary: Path) -> None:
    if temporary.exists():
        shutil.rmtree(temporary, ignore_errors=True)


# -----------------------------------------------------------------------------
# JSON, validation, metrics, and file-information helpers
# -----------------------------------------------------------------------------


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_safe(value), file, indent=2, allow_nan=False)


def require_columns(data: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required columns: " + ", ".join(missing)
        )


def standardise_fips(values: pd.Series) -> pd.Series:
    return (
        values.fillna("")
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
        .str.zfill(2)
    )


def standardise_string(values: pd.Series) -> pd.Series:
    return values.fillna("Unknown").astype(str).str.strip().replace("", "Unknown")


def has_history_marker(column: str) -> bool:
    lower = column.lower()
    return any(marker in lower for marker in HISTORY_MARKERS)


def file_information(path: Path, data: pd.DataFrame, year_column: str) -> dict[str, Any]:
    stat = path.stat()
    years = pd.to_numeric(data[year_column], errors="coerce").dropna()
    return {
        "path": str(path),
        "row_count": int(len(data)),
        "column_count": int(len(data.columns)),
        "year_min": int(years.min()) if not years.empty else None,
        "year_max": int(years.max()) if not years.empty else None,
        "size_bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "columns": list(data.columns),
        "dtypes": {column: str(dtype) for column, dtype in data.dtypes.items()},
    }


def regression_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    actual_array = np.asarray(actual, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)

    valid = np.isfinite(actual_array) & np.isfinite(prediction_array)
    if sample_weight is not None:
        weight_array = np.asarray(sample_weight, dtype=float)
        valid &= np.isfinite(weight_array) & (weight_array >= 0)
        weight_array = weight_array[valid]
        if float(weight_array.sum()) <= 0:
            weight_array = None
    else:
        weight_array = None

    actual_array = actual_array[valid]
    prediction_array = prediction_array[valid]

    if len(actual_array) == 0:
        raise ValueError("No finite rows were available for metric calculation.")

    mae = mean_absolute_error(
        actual_array,
        prediction_array,
        sample_weight=weight_array,
    )
    rmse = math.sqrt(
        mean_squared_error(
            actual_array,
            prediction_array,
            sample_weight=weight_array,
        )
    )

    if len(actual_array) < 2 or float(np.nanvar(actual_array)) == 0:
        r2 = float("nan")
    else:
        r2 = r2_score(
            actual_array,
            prediction_array,
            sample_weight=weight_array,
        )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "rows": int(len(actual_array)),
        "weight_sum": (
            float(weight_array.sum()) if weight_array is not None else None
        ),
    }


def metrics_by_year(
    table: pd.DataFrame,
    actual_column: str,
    prediction_column: str,
    weight_column: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for year, group in table.groupby("year", sort=True):
        weights = (
            group[weight_column].to_numpy(dtype=float)
            if weight_column is not None and weight_column in group.columns
            else None
        )
        rows.append(
            {
                "year": int(year),
                **regression_metrics(
                    group[actual_column].to_numpy(dtype=float),
                    group[prediction_column].to_numpy(dtype=float),
                    weights,
                ),
            }
        )
    return rows


def derive_prediction_cap(values: pd.Series, minimum: float = 1.0) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().clip(lower=0)
    if clean.empty:
        return minimum
    maximum = float(clean.max())
    high_quantile = float(clean.quantile(0.9995))
    return float(max(minimum, maximum * 2.0, high_quantile * 5.0))


def validate_and_clip_predictions(
    values: np.ndarray,
    lower: float,
    upper: float,
    label: str,
) -> np.ndarray:
    prediction = np.asarray(values, dtype=float)
    if not np.isfinite(prediction).all():
        bad = int((~np.isfinite(prediction)).sum())
        raise ValueError(f"{label} produced {bad} nonfinite predictions.")

    if upper <= lower:
        raise ValueError(f"Invalid prediction limits for {label}: {lower}, {upper}")

    implausible = prediction > upper * 100.0
    if implausible.any():
        raise ValueError(
            f"{label} produced {int(implausible.sum())} predictions more than "
            f"100 times the training-derived sanity cap ({upper})."
        )

    return np.clip(prediction, lower, upper)


def best_iteration(
    model: CatBoostClassifier | CatBoostRegressor,
    fallback: int,
) -> int:
    value = model.get_best_iteration()
    if value is None or int(value) < 0:
        return int(fallback)
    return int(value) + 1


def verify_arrays_match(
    expected: np.ndarray,
    actual: np.ndarray,
    label: str,
    rtol: float = 1e-7,
    atol: float = 1e-8,
) -> None:
    if not np.allclose(expected, actual, rtol=rtol, atol=atol, equal_nan=False):
        difference = float(np.max(np.abs(expected - actual)))
        raise ValueError(
            f"Reload verification failed for {label}. Maximum difference: {difference}"
        )


def print_metric_summary(label: str, metrics: dict[str, Any]) -> None:
    print(
        f"{label}: MAE={metrics['mae']:.6f}, "
        f"RMSE={metrics['rmse']:.6f}, R2={metrics['r2']:.6f}"
    )


def select_quick_rows_by_year(
    data: pd.DataFrame,
    year_column: str,
    rows_per_year: int,
    extra_positive_target: str | None = None,
    positive_rows_per_year: int = 0,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for year, group in data.groupby(year_column, sort=True):
        general = group.sample(
            n=min(rows_per_year, len(group)),
            random_state=RANDOM_SEED + int(year),
        )
        pieces.append(general)

        if extra_positive_target is not None and positive_rows_per_year > 0:
            positive = group[
                pd.to_numeric(group[extra_positive_target], errors="coerce") > 0
            ]
            if not positive.empty:
                pieces.append(
                    positive.sample(
                        n=min(positive_rows_per_year, len(positive)),
                        random_state=RANDOM_SEED + int(year) + 100,
                    )
                )

    sample = pd.concat(pieces, axis=0)
    sample = sample.loc[~sample.index.duplicated(keep="first")].copy()
    return sample.sort_values(year_column).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Electricity feature construction and modelling
# -----------------------------------------------------------------------------


def electricity_feature_forbidden(column: str) -> bool:
    lower = column.lower()
    exact = {
        "sequence_id",
        "state_name",
        "target_year",
        SAIDI,
        SAIFI,
        CUSTOMERS,
        "month",
    }
    if column in exact:
        return True
    if lower.startswith("target_") or lower.startswith("actual_"):
        return True
    if "saidi" in lower and not has_history_marker(lower):
        return True
    if "saifi" in lower and not has_history_marker(lower):
        return True
    if "sample_weight" in lower:
        return True
    return False


def make_one_electricity_row_per_sequence(
    electricity: pd.DataFrame,
) -> pd.DataFrame:
    required = [
        "sequence_id",
        "month",
        "utility_number",
        "state_fips",
        "state_name",
        "target_year",
        "ownership",
        "reporting_standard",
        SAIDI,
        SAIFI,
        CUSTOMERS,
        *ELECTRICITY_MONTHLY_COLUMNS,
    ]
    require_columns(electricity, required, "Electricity CSV")

    rows_per_sequence = electricity.groupby("sequence_id", sort=False).size()
    months_per_sequence = electricity.groupby("sequence_id", sort=False)[
        "month"
    ].nunique()

    if not rows_per_sequence.eq(12).all():
        count = int((~rows_per_sequence.eq(12)).sum())
        raise ValueError(
            f"{count} electricity sequences do not contain exactly 12 rows."
        )
    if not months_per_sequence.eq(12).all():
        count = int((~months_per_sequence.eq(12)).sum())
        raise ValueError(
            f"{count} electricity sequences do not contain 12 unique months."
        )
    if electricity.duplicated(["sequence_id", "month"]).any():
        raise ValueError("The electricity CSV contains duplicate sequence_id/month rows.")

    annual_columns = [
        column
        for column in electricity.columns
        if column not in set(ELECTRICITY_MONTHLY_COLUMNS + ["month"])
    ]

    changing_columns: list[str] = []
    grouped = electricity.groupby("sequence_id", sort=False)
    for column in annual_columns:
        if column == "sequence_id":
            continue
        if grouped[column].nunique(dropna=False).gt(1).any():
            changing_columns.append(column)

    if changing_columns:
        raise ValueError(
            "Columns not declared as monthly change inside electricity sequences: "
            + ", ".join(changing_columns)
        )

    annual = electricity[annual_columns].drop_duplicates("sequence_id").copy()
    monthly = electricity.pivot(
        index="sequence_id",
        columns="month",
        values=ELECTRICITY_MONTHLY_COLUMNS,
    )
    monthly.columns = [
        f"{column}_month_{int(month):02d}" for column, month in monthly.columns
    ]
    annual = annual.merge(
        monthly.reset_index(),
        on="sequence_id",
        how="left",
        validate="one_to_one",
    )
    return annual


def prepare_electricity_features(
    annual: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    annual = annual.copy()
    annual["target_year"] = pd.to_numeric(
        annual["target_year"], errors="coerce"
    ).astype("Int64")
    annual[SAIDI] = pd.to_numeric(annual[SAIDI], errors="coerce")
    annual[SAIFI] = pd.to_numeric(annual[SAIFI], errors="coerce")
    annual[CUSTOMERS] = pd.to_numeric(annual[CUSTOMERS], errors="coerce")
    annual["state_fips"] = standardise_fips(annual["state_fips"])

    annual.dropna(subset=["target_year"], inplace=True)
    annual["target_year"] = annual["target_year"].astype(int)

    for column in ELECTRICITY_CATEGORICAL_COLUMNS:
        annual[column] = standardise_string(annual[column])

    numeric_features: list[str] = []
    rejected: list[str] = []

    for column in annual.columns:
        if column in ELECTRICITY_CATEGORICAL_COLUMNS:
            continue
        if electricity_feature_forbidden(column):
            rejected.append(column)
            continue

        converted = pd.to_numeric(annual[column], errors="coerce")
        original_nonmissing = int(annual[column].notna().sum())
        converted_nonmissing = int(converted.notna().sum())
        if original_nonmissing == 0 or converted_nonmissing >= max(
            1, int(original_nonmissing * 0.95)
        ):
            annual[column] = converted.astype(np.float32)
            numeric_features.append(column)
        else:
            rejected.append(column)

    features = numeric_features + ELECTRICITY_CATEGORICAL_COLUMNS
    forbidden_selected = [
        column for column in features if electricity_feature_forbidden(column)
    ]
    if forbidden_selected:
        raise ValueError(
            "Electricity leakage audit failed: " + ", ".join(forbidden_selected)
        )

    return annual, features, numeric_features, rejected


def electricity_frame(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for column in features:
        if column in ELECTRICITY_CATEGORICAL_COLUMNS:
            columns[column] = standardise_string(data[column])
        else:
            columns[column] = pd.to_numeric(
                data[column], errors="coerce"
            ).astype(np.float32)
    return pd.DataFrame(columns, index=data.index)[features]


def electricity_weights(data: pd.DataFrame) -> np.ndarray:
    weights = (
        pd.to_numeric(data[CUSTOMERS], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=float)
    )
    mean = float(weights.mean()) if len(weights) else 0.0
    if mean <= 0:
        return np.ones(len(data), dtype=float)
    return weights / mean


def electricity_pool(
    data: pd.DataFrame,
    features: list[str],
    label: np.ndarray | None = None,
    weight: np.ndarray | None = None,
) -> Pool:
    return Pool(
        data=electricity_frame(data, features),
        label=label,
        weight=weight,
        cat_features=ELECTRICITY_CATEGORICAL_COLUMNS,
    )


def electricity_regressor_parameters(
    iterations: int,
    depth: int,
    threads: int,
    quick: bool,
    early_stopping: bool,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "iterations": iterations,
        "depth": depth,
        "learning_rate": 0.03,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "l2_leaf_reg": 7,
        "random_seed": RANDOM_SEED,
        "has_time": True,
        "allow_writing_files": False,
        "thread_count": threads,
        "verbose": False if quick else 100,
    }
    if early_stopping:
        parameters.update({"od_type": "Iter", "od_wait": 30 if quick else 150})
    return parameters


def electricity_classifier_parameters(
    iterations: int,
    threads: int,
    quick: bool,
    early_stopping: bool,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "iterations": iterations,
        "depth": 6,
        "learning_rate": 0.03,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "l2_leaf_reg": 7,
        "random_seed": RANDOM_SEED,
        "has_time": True,
        "allow_writing_files": False,
        "thread_count": threads,
        "verbose": False if quick else 100,
    }
    if early_stopping:
        parameters.update({"od_type": "Iter", "od_wait": 30 if quick else 150})
    return parameters


def tune_electricity_regressor(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    target: str,
    depth: int,
    threads: int,
    quick: bool,
) -> tuple[CatBoostRegressor, int]:
    maximum_iterations = 180 if quick else 2_000
    train = train.dropna(subset=[target]).copy()
    validation = validation.dropna(subset=[target]).copy()
    if train.empty or validation.empty:
        raise ValueError(f"Empty chronological split while tuning {target}.")

    model = CatBoostRegressor(
        **electricity_regressor_parameters(
            maximum_iterations, depth, threads, quick, early_stopping=True
        )
    )
    model.fit(
        electricity_pool(
            train,
            features,
            label=np.log1p(train[target].to_numpy(dtype=float)),
            weight=electricity_weights(train),
        ),
        eval_set=electricity_pool(
            validation,
            features,
            label=np.log1p(validation[target].to_numpy(dtype=float)),
            weight=electricity_weights(validation),
        ),
        use_best_model=True,
    )
    return model, best_iteration(model, maximum_iterations)


def tune_saidi_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    threads: int,
    quick: bool,
) -> tuple[dict[str, Any], dict[str, int], float]:
    train = train.dropna(subset=[SAIDI]).copy()
    validation = validation.dropna(subset=[SAIDI]).copy()
    if train.empty or validation.empty:
        raise ValueError("SAIDI train or validation split is empty.")

    threshold = float(train[SAIDI].quantile(0.90))
    train["_high_saidi"] = (train[SAIDI] > threshold).astype(int)
    validation["_high_saidi"] = (validation[SAIDI] > threshold).astype(int)

    if train["_high_saidi"].nunique() != 2:
        raise ValueError("SAIDI classifier training data does not contain both classes.")
    if validation["_high_saidi"].nunique() != 2:
        raise ValueError(
            "SAIDI validation data does not contain both high and normal events."
        )

    classifier_max = 160 if quick else 1_500
    classifier = CatBoostClassifier(
        **electricity_classifier_parameters(
            classifier_max, threads, quick, early_stopping=True
        )
    )
    classifier.fit(
        electricity_pool(
            train,
            features,
            label=train["_high_saidi"].to_numpy(dtype=int),
            weight=electricity_weights(train),
        ),
        eval_set=electricity_pool(
            validation,
            features,
            label=validation["_high_saidi"].to_numpy(dtype=int),
            weight=electricity_weights(validation),
        ),
        use_best_model=True,
    )

    normal_model, normal_iterations = tune_electricity_regressor(
        train[train["_high_saidi"] == 0],
        validation[validation["_high_saidi"] == 0],
        features,
        SAIDI,
        depth=7,
        threads=threads,
        quick=quick,
    )
    high_model, high_iterations = tune_electricity_regressor(
        train[train["_high_saidi"] == 1],
        validation[validation["_high_saidi"] == 1],
        features,
        SAIDI,
        depth=5,
        threads=threads,
        quick=quick,
    )

    models = {
        "classifier": classifier,
        "normal": normal_model,
        "high": high_model,
    }
    iterations = {
        "classifier": best_iteration(classifier, classifier_max),
        "normal": normal_iterations,
        "high": high_iterations,
    }
    return models, iterations, threshold


def fit_saidi_models(
    data: pd.DataFrame,
    features: list[str],
    iterations: dict[str, int],
    threads: int,
    quick: bool,
) -> tuple[dict[str, Any], float]:
    fit_data = data.dropna(subset=[SAIDI]).copy()
    if fit_data.empty:
        raise ValueError("No SAIDI rows were available for refitting.")

    threshold = float(fit_data[SAIDI].quantile(0.90))
    fit_data["_high_saidi"] = (fit_data[SAIDI] > threshold).astype(int)
    if fit_data["_high_saidi"].nunique() != 2:
        raise ValueError("SAIDI refit data does not contain both classes.")

    classifier = CatBoostClassifier(
        **electricity_classifier_parameters(
            iterations["classifier"], threads, quick, early_stopping=False
        )
    )
    classifier.fit(
        electricity_pool(
            fit_data,
            features,
            label=fit_data["_high_saidi"].to_numpy(dtype=int),
            weight=electricity_weights(fit_data),
        )
    )

    normal_data = fit_data[fit_data["_high_saidi"] == 0]
    high_data = fit_data[fit_data["_high_saidi"] == 1]

    normal = CatBoostRegressor(
        **electricity_regressor_parameters(
            iterations["normal"], 7, threads, quick, early_stopping=False
        )
    )
    normal.fit(
        electricity_pool(
            normal_data,
            features,
            label=np.log1p(normal_data[SAIDI].to_numpy(dtype=float)),
            weight=electricity_weights(normal_data),
        )
    )

    high = CatBoostRegressor(
        **electricity_regressor_parameters(
            iterations["high"], 5, threads, quick, early_stopping=False
        )
    )
    high.fit(
        electricity_pool(
            high_data,
            features,
            label=np.log1p(high_data[SAIDI].to_numpy(dtype=float)),
            weight=electricity_weights(high_data),
        )
    )
    return {"classifier": classifier, "normal": normal, "high": high}, threshold


def fit_saifi_model(
    data: pd.DataFrame,
    features: list[str],
    iterations: int,
    threads: int,
    quick: bool,
) -> CatBoostRegressor:
    fit_data = data.dropna(subset=[SAIFI]).copy()
    if fit_data.empty:
        raise ValueError("No SAIFI rows were available for refitting.")
    model = CatBoostRegressor(
        **electricity_regressor_parameters(
            iterations, 7, threads, quick, early_stopping=False
        )
    )
    model.fit(
        electricity_pool(
            fit_data,
            features,
            label=np.log1p(fit_data[SAIFI].to_numpy(dtype=float)),
            weight=electricity_weights(fit_data),
        )
    )
    return model


def predict_saidi(
    models: dict[str, Any],
    data: pd.DataFrame,
    features: list[str],
    prediction_cap: float,
) -> np.ndarray:
    pool = electricity_pool(data, features)
    probability = models["classifier"].predict_proba(pool)[:, 1]
    maximum_log = math.log1p(prediction_cap)
    normal = np.expm1(
        np.clip(models["normal"].predict(pool), 0.0, maximum_log)
    )
    high = np.expm1(np.clip(models["high"].predict(pool), 0.0, maximum_log))
    blended = (1.0 - probability) * normal + probability * high
    return validate_and_clip_predictions(
        blended, 0.0, prediction_cap, "SAIDI two-stage CatBoost"
    )


def predict_saifi(
    model: CatBoostRegressor,
    data: pd.DataFrame,
    features: list[str],
    prediction_cap: float,
) -> np.ndarray:
    pool = electricity_pool(data, features)
    maximum_log = math.log1p(prediction_cap)
    prediction = np.expm1(np.clip(model.predict(pool), 0.0, maximum_log))
    return validate_and_clip_predictions(
        prediction, 0.0, prediction_cap, "SAIFI log-CatBoost"
    )


def aggregate_electricity_state_year(
    data: pd.DataFrame,
    predictions: np.ndarray,
    target: str,
    short_name: str,
) -> pd.DataFrame:
    table = data[
        ["state_fips", "state_name", "target_year", target, CUSTOMERS]
    ].copy()
    table["prediction"] = predictions
    table = table.dropna(subset=[target]).copy()
    table[CUSTOMERS] = (
        pd.to_numeric(table[CUSTOMERS], errors="coerce").fillna(0).clip(lower=0)
    )
    table["actual_weighted"] = table[target] * table[CUSTOMERS]
    table["prediction_weighted"] = table["prediction"] * table[CUSTOMERS]

    grouped = table.groupby(
        ["state_fips", "state_name", "target_year"],
        as_index=False,
        dropna=False,
    )[["actual_weighted", "prediction_weighted", CUSTOMERS]].sum()
    grouped = grouped[grouped[CUSTOMERS] > 0].copy()
    grouped[f"actual_{short_name}"] = grouped["actual_weighted"] / grouped[CUSTOMERS]
    grouped[f"predicted_{short_name}"] = (
        grouped["prediction_weighted"] / grouped[CUSTOMERS]
    )
    grouped.rename(columns={"target_year": "year", CUSTOMERS: "metric_weight"}, inplace=True)
    return grouped[
        [
            "state_fips",
            "state_name",
            "year",
            f"actual_{short_name}",
            f"predicted_{short_name}",
            "metric_weight",
        ]
    ]


def aggregate_electricity_actuals(
    annual: pd.DataFrame,
    target: str,
    short_name: str,
) -> pd.DataFrame:
    eligible = annual.dropna(subset=[target]).copy()
    placeholder = np.zeros(len(eligible), dtype=float)
    table = aggregate_electricity_state_year(
        eligible, placeholder, target, short_name
    )
    return table[
        ["state_fips", "state_name", "year", f"actual_{short_name}"]
    ]


def electricity_metric_bundle(
    data: pd.DataFrame,
    predictions: np.ndarray,
    target: str,
    short_name: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    valid = data[target].notna().to_numpy()
    utility = regression_metrics(
        data.loc[valid, target].to_numpy(dtype=float),
        predictions[valid],
        electricity_weights(data.loc[valid]),
    )
    state = aggregate_electricity_state_year(
        data.loc[valid], predictions[valid], target, short_name
    )
    state_metrics = regression_metrics(
        state[f"actual_{short_name}"].to_numpy(dtype=float),
        state[f"predicted_{short_name}"].to_numpy(dtype=float),
        state["metric_weight"].to_numpy(dtype=float),
    )
    return {"utility_row": utility, "state_year": state_metrics}, state


def catboost_factors(
    models: Iterable[CatBoostClassifier | CatBoostRegressor],
    features: list[str],
) -> list[str]:
    importance = [
        np.asarray(model.get_feature_importance(), dtype=float) for model in models
    ]
    average = np.mean(importance, axis=0)
    table = pd.DataFrame({"feature": features, "importance": average})
    ignored = ("utility_number", "state_fips", "available_flag", "mapping_available")
    for word in ignored:
        table = table[~table["feature"].str.contains(word, case=False, na=False)]
    table = table.sort_values("importance", ascending=False)
    factors = table["feature"].head(5).tolist()
    if not factors:
        factors = [column for column in features if column not in ELECTRICITY_CATEGORICAL_COLUMNS][:5]
    return factors


def save_and_verify_electricity_models(
    output: Path,
    saidi_models: dict[str, Any],
    saifi_model: CatBoostRegressor,
    sample: pd.DataFrame,
    features: list[str],
    saidi_cap: float,
    saifi_cap: float,
) -> None:
    paths = {
        "classifier": output / "saidi_high_event_classifier.cbm",
        "normal": output / "saidi_normal_regressor.cbm",
        "high": output / "saidi_high_regressor.cbm",
    }
    saidi_models["classifier"].save_model(str(paths["classifier"]))
    saidi_models["normal"].save_model(str(paths["normal"]))
    saidi_models["high"].save_model(str(paths["high"]))
    saifi_path = output / "saifi_log_catboost.cbm"
    saifi_model.save_model(str(saifi_path))

    expected_saidi = predict_saidi(saidi_models, sample, features, saidi_cap)
    expected_saifi = predict_saifi(saifi_model, sample, features, saifi_cap)

    loaded_saidi = {
        "classifier": CatBoostClassifier(),
        "normal": CatBoostRegressor(),
        "high": CatBoostRegressor(),
    }
    loaded_saidi["classifier"].load_model(str(paths["classifier"]))
    loaded_saidi["normal"].load_model(str(paths["normal"]))
    loaded_saidi["high"].load_model(str(paths["high"]))
    loaded_saifi = CatBoostRegressor()
    loaded_saifi.load_model(str(saifi_path))

    actual_saidi = predict_saidi(loaded_saidi, sample, features, saidi_cap)
    actual_saifi = predict_saifi(loaded_saifi, sample, features, saifi_cap)
    verify_arrays_match(expected_saidi, actual_saidi, "SAIDI CatBoost artifacts")
    verify_arrays_match(expected_saifi, actual_saifi, "SAIFI CatBoost artifact")


def train_electricity_pipeline(
    input_path: Path,
    output: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    print()
    print("=" * 72)
    print("ELECTRICITY PIPELINES")
    print("=" * 72)
    print("Reading:", input_path)

    monthly = pd.read_csv(input_path, low_memory=False)
    input_info = file_information(input_path, monthly, "target_year")
    annual = make_one_electricity_row_per_sequence(monthly)
    del monthly
    gc.collect()

    annual, features, numeric_features, rejected_features = prepare_electricity_features(annual)
    annual.sort_values(["target_year", "sequence_id"], inplace=True)
    annual.reset_index(drop=True, inplace=True)

    if args.quick:
        quick_parts = []
        for year, group in annual.groupby("target_year", sort=True):
            quick_parts.append(
                group.sample(
                    n=min(150, len(group)),
                    random_state=RANDOM_SEED + int(year),
                )
            )
        annual = pd.concat(quick_parts, ignore_index=True)
        print("QUICK MODE: electricity sequences reduced to", len(annual))

    train = annual[annual["target_year"] <= TRAIN_END_YEAR]
    validation = annual[annual["target_year"].isin(VALIDATION_YEARS)]
    backtest = annual[annual["target_year"].isin(BACKTEST_YEARS)]
    through_validation = annual[annual["target_year"] <= VALIDATION_END_YEAR]

    if train.empty or validation.empty or backtest.empty:
        raise ValueError("One or more electricity chronological splits are empty.")

    saidi_cap = derive_prediction_cap(train[SAIDI], minimum=1.0)
    saifi_cap = derive_prediction_cap(train[SAIFI], minimum=0.1)

    print("Monthly input rows:", input_info["row_count"])
    print("Annual sequences used:", len(annual))
    print("Electricity features:", len(features))

    selection_saidi_models, saidi_iterations, selection_threshold = tune_saidi_models(
        train, validation, features, args.threads, args.quick
    )
    selection_saifi_model, saifi_iterations = tune_electricity_regressor(
        train,
        validation,
        features,
        SAIFI,
        depth=7,
        threads=args.threads,
        quick=args.quick,
    )

    validation_saidi_prediction = predict_saidi(
        selection_saidi_models, validation, features, saidi_cap
    )
    validation_saifi_prediction = predict_saifi(
        selection_saifi_model, validation, features, saifi_cap
    )
    validation_saidi_metrics, validation_saidi_state = electricity_metric_bundle(
        validation, validation_saidi_prediction, SAIDI, "saidi"
    )
    validation_saifi_metrics, validation_saifi_state = electricity_metric_bundle(
        validation, validation_saifi_prediction, SAIFI, "saifi"
    )
    print_metric_summary(
        "SAIDI validation state-year", validation_saidi_metrics["state_year"]
    )
    print_metric_summary(
        "SAIFI validation state-year", validation_saifi_metrics["state_year"]
    )

    del selection_saidi_models
    del selection_saifi_model
    gc.collect()

    evaluation_saidi_models, evaluation_threshold = fit_saidi_models(
        through_validation,
        features,
        saidi_iterations,
        args.threads,
        args.quick,
    )
    evaluation_saifi_model = fit_saifi_model(
        through_validation,
        features,
        saifi_iterations,
        args.threads,
        args.quick,
    )

    backtest_saidi_prediction = predict_saidi(
        evaluation_saidi_models, backtest, features, saidi_cap
    )
    backtest_saifi_prediction = predict_saifi(
        evaluation_saifi_model, backtest, features, saifi_cap
    )
    backtest_saidi_metrics, backtest_saidi_state = electricity_metric_bundle(
        backtest, backtest_saidi_prediction, SAIDI, "saidi"
    )
    backtest_saifi_metrics, backtest_saifi_state = electricity_metric_bundle(
        backtest, backtest_saifi_prediction, SAIFI, "saifi"
    )
    print_metric_summary(
        "SAIDI untouched backtest state-year", backtest_saidi_metrics["state_year"]
    )
    print_metric_summary(
        "SAIFI untouched backtest state-year", backtest_saifi_metrics["state_year"]
    )

    saidi_actual = aggregate_electricity_actuals(annual, SAIDI, "saidi")
    saifi_actual = aggregate_electricity_actuals(annual, SAIFI, "saifi")
    saidi_signal = pd.concat(
        [
            validation_saidi_state[
                ["state_fips", "state_name", "year", "predicted_saidi"]
            ],
            backtest_saidi_state[
                ["state_fips", "state_name", "year", "predicted_saidi"]
            ],
        ],
        ignore_index=True,
    )
    saifi_signal = pd.concat(
        [
            validation_saifi_state[
                ["state_fips", "state_name", "year", "predicted_saifi"]
            ],
            backtest_saifi_state[
                ["state_fips", "state_name", "year", "predicted_saifi"]
            ],
        ],
        ignore_index=True,
    )

    electricity_history = saidi_actual.merge(
        saidi_signal,
        on=["state_fips", "state_name", "year"],
        how="left",
        validate="one_to_one",
    )
    electricity_history = electricity_history.merge(
        saifi_actual,
        on=["state_fips", "state_name", "year"],
        how="outer",
        validate="one_to_one",
    )
    electricity_history = electricity_history.merge(
        saifi_signal,
        on=["state_fips", "state_name", "year"],
        how="left",
        validate="one_to_one",
    )
    electricity_history.sort_values(["state_fips", "year"], inplace=True)
    electricity_history.to_csv(output / "electricity_state_history.csv", index=False)

    del evaluation_saidi_models
    del evaluation_saifi_model
    gc.collect()

    deployment_saidi_models, deployment_threshold = fit_saidi_models(
        annual,
        features,
        saidi_iterations,
        args.threads,
        args.quick,
    )
    deployment_saifi_model = fit_saifi_model(
        annual,
        features,
        saifi_iterations,
        args.threads,
        args.quick,
    )

    verification_sample = annual.sample(
        n=min(32, len(annual)), random_state=RANDOM_SEED
    ).copy()
    save_and_verify_electricity_models(
        output,
        deployment_saidi_models,
        deployment_saifi_model,
        verification_sample,
        features,
        saidi_cap,
        saifi_cap,
    )

    factors = {
        "saidi": catboost_factors(
            [
                deployment_saidi_models["classifier"],
                deployment_saidi_models["normal"],
                deployment_saidi_models["high"],
            ],
            features,
        ),
        "saifi": catboost_factors([deployment_saifi_model], features),
    }

    information = {
        "target_columns": {"saidi": SAIDI, "saifi": SAIFI},
        "features": features,
        "numeric_features": numeric_features,
        "categorical_features": ELECTRICITY_CATEGORICAL_COLUMNS,
        "rejected_columns": rejected_features,
        "iterations": {
            "saidi": saidi_iterations,
            "saifi": saifi_iterations,
        },
        "high_saidi_thresholds": {
            "selection_train_through_2018": selection_threshold,
            "evaluation_refit_through_2020": evaluation_threshold,
            "deployment_refit_all_observed": deployment_threshold,
        },
        "prediction_caps_from_selection_training": {
            "saidi": saidi_cap,
            "saifi": saifi_cap,
        },
        "deployment_year_range": [
            int(annual["target_year"].min()),
            int(annual["target_year"].max()),
        ],
        "model_files": {
            "saidi_classifier": "saidi_high_event_classifier.cbm",
            "saidi_normal": "saidi_normal_regressor.cbm",
            "saidi_high": "saidi_high_regressor.cbm",
            "saifi": "saifi_log_catboost.cbm",
        },
    }
    write_json(output / "electricity_model_information.json", information)

    metrics = {
        "saidi": {
            "validation": validation_saidi_metrics,
            "backtest": backtest_saidi_metrics,
            "backtest_by_year": metrics_by_year(
                backtest_saidi_state,
                "actual_saidi",
                "predicted_saidi",
                "metric_weight",
            ),
        },
        "saifi": {
            "validation": validation_saifi_metrics,
            "backtest": backtest_saifi_metrics,
            "backtest_by_year": metrics_by_year(
                backtest_saifi_state,
                "actual_saifi",
                "predicted_saifi",
                "metric_weight",
            ),
        },
    }

    schema = {
        "ordered_features": features,
        "numeric_features": numeric_features,
        "categorical_features": ELECTRICITY_CATEGORICAL_COLUMNS,
        "input_dtypes": input_info["dtypes"],
        "targets": [SAIDI, SAIFI],
        "selection_training_years": f"<= {TRAIN_END_YEAR}",
        "validation_years": list(VALIDATION_YEARS),
        "backtest_years": list(BACKTEST_YEARS),
        "deployment_year_range": information["deployment_year_range"],
        "prediction_caps": information["prediction_caps_from_selection_training"],
    }

    del annual
    del deployment_saidi_models
    del deployment_saifi_model
    gc.collect()

    return {
        "history": electricity_history,
        "metrics": metrics,
        "schema": schema,
        "factors": factors,
        "input_information": input_info,
        "model_information": information,
    }


# -----------------------------------------------------------------------------
# Drought: ANOVA SelectKBest plus Ridge
# -----------------------------------------------------------------------------


def drought_feature_forbidden(column: str) -> bool:
    lower = column.lower()
    exact = {
        "county_fips",
        "state_fips",
        "state_abbreviation",
        "state_name",
        "year",
        "map_date",
        DROUGHT,
        DROUGHT_WEIGHT,
        "usdm_cumulative_order_valid",
    }
    if column in exact:
        return True
    if lower.startswith("target_") or lower.startswith("actual_"):
        return True
    if "current" in lower and ("drought" in lower or "usdm" in lower):
        return True
    target_component_tokens = (
        "d0_percent",
        "d1_percent",
        "d2_percent",
        "d3_percent",
        "d4_percent",
        "d0_area",
        "d1_area",
        "d2_area",
        "d3_area",
        "d4_area",
    )
    if any(token in lower for token in target_component_tokens) and not has_history_marker(lower):
        return True
    return False


def prepare_drought_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    required = [
        "county_fips",
        "state_fips",
        "state_abbreviation",
        "year",
        "map_date",
        DROUGHT,
        DROUGHT_WEIGHT,
    ]
    require_columns(data, required, "Drought CSV")

    if data.duplicated(["county_fips", "map_date"]).any():
        raise ValueError("The drought CSV contains duplicate county_fips/map_date rows.")

    data = data.copy()
    data["state_fips"] = standardise_fips(data["state_fips"])
    data["state_abbreviation"] = standardise_string(data["state_abbreviation"])
    data["year"] = pd.to_numeric(data["year"], errors="coerce").astype("Int64")
    data[DROUGHT] = pd.to_numeric(data[DROUGHT], errors="coerce")
    data[DROUGHT_WEIGHT] = pd.to_numeric(data[DROUGHT_WEIGHT], errors="coerce")
    data["map_date"] = data["map_date"].astype(str)
    data.dropna(subset=["year", DROUGHT], inplace=True)
    data["year"] = data["year"].astype(int)

    if ((data[DROUGHT] < 0) | (data[DROUGHT] > 100)).any():
        raise ValueError("Drought target values must be between 0 and 100.")

    numeric_features: list[str] = []
    rejected: list[str] = []
    for column in data.columns:
        if drought_feature_forbidden(column):
            rejected.append(column)
            continue
        if pd.api.types.is_numeric_dtype(data[column]):
            numeric_features.append(column)

    if not numeric_features:
        raise ValueError("No leakage-safe numeric drought features were found.")
    return data, numeric_features, rejected


def drought_candidate_feature_names(
    numeric_features: list[str], state_categories: list[str]
) -> list[str]:
    return numeric_features + [f"state_abbreviation=={state}" for state in state_categories]


def drought_full_feature_matrix(
    data: pd.DataFrame,
    numeric_features: list[str],
    state_categories: list[str],
) -> np.ndarray:
    numeric = np.column_stack(
        [
            pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=np.float32)
            for column in numeric_features
        ]
    )
    states = data["state_abbreviation"].astype(str).to_numpy()
    one_hot = np.column_stack(
        [(states == state).astype(np.float32) for state in state_categories]
    )
    return np.hstack([numeric, one_hot]).astype(np.float32, copy=False)


def drought_selected_feature_matrix(
    data: pd.DataFrame,
    selected_features: list[str],
) -> np.ndarray:
    columns: list[np.ndarray] = []
    states = data["state_abbreviation"].astype(str).to_numpy()
    prefix = "state_abbreviation=="
    for feature in selected_features:
        if feature.startswith(prefix):
            state = feature[len(prefix) :]
            columns.append((states == state).astype(np.float32))
        else:
            columns.append(
                pd.to_numeric(data[feature], errors="coerce").to_numpy(
                    dtype=np.float32
                )
            )
    return np.column_stack(columns).astype(np.float32, copy=False)


def select_drought_features(
    train: pd.DataFrame,
    numeric_features: list[str],
    state_categories: list[str],
    quick: bool,
) -> tuple[list[str], dict[str, Any]]:
    maximum_rows = 50_000 if quick else 500_000
    sample = train.sample(
        n=min(maximum_rows, len(train)), random_state=RANDOM_SEED
    ).copy()
    all_feature_names = drought_candidate_feature_names(
        numeric_features, state_categories
    )
    matrix = drought_full_feature_matrix(sample, numeric_features, state_categories)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    transformed = imputer.fit_transform(matrix)
    transformed = scaler.fit_transform(transformed)
    selector = SelectKBest(
        score_func=f_regression, k=min(20, len(all_feature_names))
    )
    selector.fit(transformed, sample[DROUGHT].to_numpy(dtype=float))
    selected = np.asarray(all_feature_names)[selector.get_support()].tolist()

    audit = {
        "selection_rows": int(len(sample)),
        "all_candidate_features": all_feature_names,
        "selected_features": selected,
        "selection_imputer": imputer,
        "selection_scaler": scaler,
        "selector": selector,
    }
    del matrix
    del transformed
    del sample
    gc.collect()
    return selected, audit


def fit_drought_ridge(
    data: pd.DataFrame,
    selected_features: list[str],
) -> dict[str, Any]:
    matrix = drought_selected_feature_matrix(data, selected_features)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    transformed = imputer.fit_transform(matrix)
    transformed = scaler.fit_transform(transformed)
    model = Ridge(alpha=1.0)
    model.fit(
        transformed,
        data[DROUGHT].to_numpy(dtype=float),
        sample_weight=drought_weights(data),
    )
    del matrix
    del transformed
    gc.collect()
    return {"imputer": imputer, "scaler": scaler, "model": model}


def drought_weights(data: pd.DataFrame) -> np.ndarray:
    weights = (
        pd.to_numeric(data[DROUGHT_WEIGHT], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=float)
    )
    mean = float(weights.mean()) if len(weights) else 0.0
    if mean <= 0:
        return np.ones(len(data), dtype=float)
    return weights / mean


def predict_drought_model(
    artifact: dict[str, Any],
    data: pd.DataFrame,
    selected_features: list[str],
) -> np.ndarray:
    matrix = drought_selected_feature_matrix(data, selected_features)
    transformed = artifact["imputer"].transform(matrix)
    transformed = artifact["scaler"].transform(transformed)
    raw = artifact["model"].predict(transformed)
    del matrix
    del transformed
    gc.collect()
    return validate_and_clip_predictions(raw, 0.0, 100.0, "Drought Ridge")


def aggregate_drought_state_year(
    data: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    table = data[
        [
            "state_fips",
            "state_abbreviation",
            "year",
            "map_date",
            DROUGHT,
            DROUGHT_WEIGHT,
        ]
    ].copy()
    table["prediction"] = predictions
    table["weight"] = (
        pd.to_numeric(table[DROUGHT_WEIGHT], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    table["actual_weighted"] = table[DROUGHT] * table["weight"]
    table["prediction_weighted"] = table["prediction"] * table["weight"]

    weekly = table.groupby(
        ["state_fips", "state_abbreviation", "year", "map_date"],
        as_index=False,
        dropna=False,
    )[["actual_weighted", "prediction_weighted", "weight"]].sum()
    weekly = weekly[weekly["weight"] > 0].copy()
    weekly["weekly_actual"] = weekly["actual_weighted"] / weekly["weight"]
    weekly["weekly_prediction"] = weekly["prediction_weighted"] / weekly["weight"]

    state_year = weekly.groupby(
        ["state_fips", "state_abbreviation", "year"],
        as_index=False,
        dropna=False,
    )[["weekly_actual", "weekly_prediction"]].mean()
    state_year.rename(
        columns={
            "weekly_actual": "actual_drought",
            "weekly_prediction": "predicted_drought",
        },
        inplace=True,
    )
    state_year["metric_weight"] = 1.0
    return state_year


def aggregate_drought_actuals(data: pd.DataFrame) -> pd.DataFrame:
    state = aggregate_drought_state_year(
        data, np.zeros(len(data), dtype=float)
    )
    return state[
        ["state_fips", "state_abbreviation", "year", "actual_drought"]
    ]


def drought_metric_bundle(
    data: pd.DataFrame,
    predictions: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    row_metrics = regression_metrics(
        data[DROUGHT].to_numpy(dtype=float),
        predictions,
        drought_weights(data),
    )
    state = aggregate_drought_state_year(data, predictions)
    state_metrics = regression_metrics(
        state["actual_drought"].to_numpy(dtype=float),
        state["predicted_drought"].to_numpy(dtype=float),
    )
    return {"county_week_row": row_metrics, "state_year": state_metrics}, state


def save_and_verify_drought_artifact(
    output: Path,
    artifact: dict[str, Any],
    selected_features: list[str],
    selection_audit: dict[str, Any],
    sample: pd.DataFrame,
) -> None:
    path = output / "drought_anova_ridge.joblib"
    saved = {
        "model_family": MODEL_LABELS["drought"],
        "selected_features": selected_features,
        "all_candidate_features": selection_audit["all_candidate_features"],
        "selection_rows": selection_audit["selection_rows"],
        "selection_imputer": selection_audit["selection_imputer"],
        "selection_scaler": selection_audit["selection_scaler"],
        "selector": selection_audit["selector"],
        "imputer": artifact["imputer"],
        "scaler": artifact["scaler"],
        "model": artifact["model"],
        "target": DROUGHT,
        "clipping": [0.0, 100.0],
    }
    expected = predict_drought_model(saved, sample, selected_features)
    joblib.dump(saved, path, compress=3)
    loaded = joblib.load(path)
    actual = predict_drought_model(loaded, sample, selected_features)
    verify_arrays_match(expected, actual, "drought_anova_ridge.joblib")


def train_drought_pipeline(
    input_path: Path,
    output: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    print()
    print("=" * 72)
    print("DROUGHT PIPELINE")
    print("=" * 72)
    print("Reading:", input_path)

    raw = pd.read_csv(input_path, low_memory=False)
    input_info = file_information(input_path, raw, "year")
    data, numeric_features, rejected_features = prepare_drought_data(raw)
    del raw
    gc.collect()

    if args.quick:
        data = select_quick_rows_by_year(data, "year", rows_per_year=8_000)
        print("QUICK MODE: drought rows reduced to", len(data))

    observed_year_min = int(data["year"].min())
    observed_year_max = int(data["year"].max())

    train = data[data["year"] <= TRAIN_END_YEAR]
    validation = data[data["year"].isin(VALIDATION_YEARS)]
    if train.empty or validation.empty:
        raise ValueError("Drought training or validation split is empty.")

    state_categories = sorted(train["state_abbreviation"].dropna().unique().tolist())
    selected_features, selection_audit = select_drought_features(
        train, numeric_features, state_categories, args.quick
    )
    print("Drought candidate features:", len(selection_audit["all_candidate_features"]))
    print("ANOVA-selected features:", len(selected_features))

    selection_model = fit_drought_ridge(train, selected_features)
    validation_prediction = predict_drought_model(
        selection_model, validation, selected_features
    )
    validation_metrics, validation_state = drought_metric_bundle(
        validation, validation_prediction
    )
    print_metric_summary(
        "Drought validation state-year", validation_metrics["state_year"]
    )
    del selection_model
    del train
    del validation
    gc.collect()

    through_validation = data[data["year"] <= VALIDATION_END_YEAR]
    backtest = data[data["year"].isin(BACKTEST_YEARS)]
    if through_validation.empty or backtest.empty:
        raise ValueError("Drought evaluation refit or backtest split is empty.")

    evaluation_model = fit_drought_ridge(through_validation, selected_features)
    backtest_prediction = predict_drought_model(
        evaluation_model, backtest, selected_features
    )
    backtest_metrics, backtest_state = drought_metric_bundle(
        backtest, backtest_prediction
    )
    print_metric_summary(
        "Drought untouched backtest state-year", backtest_metrics["state_year"]
    )
    del through_validation
    del backtest
    gc.collect()

    actual_history = aggregate_drought_actuals(data)
    prediction_signal = pd.concat(
        [
            validation_state[
                ["state_fips", "state_abbreviation", "year", "predicted_drought"]
            ],
            backtest_state[
                ["state_fips", "state_abbreviation", "year", "predicted_drought"]
            ],
        ],
        ignore_index=True,
    )
    history = actual_history.merge(
        prediction_signal,
        on=["state_fips", "state_abbreviation", "year"],
        how="left",
        validate="one_to_one",
    )
    history.sort_values(["state_fips", "year"], inplace=True)
    history.to_csv(output / "drought_state_history.csv", index=False)

    del evaluation_model
    gc.collect()

    deployment_model = fit_drought_ridge(data, selected_features)
    verification_sample = data.sample(
        n=min(64, len(data)), random_state=RANDOM_SEED
    ).copy()
    save_and_verify_drought_artifact(
        output,
        deployment_model,
        selected_features,
        selection_audit,
        verification_sample,
    )

    coefficient_table = pd.DataFrame(
        {
            "feature": selected_features,
            "importance": np.abs(deployment_model["model"].coef_),
        }
    )
    coefficient_table = coefficient_table[
        ~coefficient_table["feature"].str.startswith("state_abbreviation==")
    ].sort_values("importance", ascending=False)
    factors = coefficient_table["feature"].head(5).tolist()
    if not factors:
        factors = selected_features[:5]

    metrics = {
        "validation": validation_metrics,
        "backtest": backtest_metrics,
        "backtest_by_year": metrics_by_year(
            backtest_state, "actual_drought", "predicted_drought"
        ),
    }
    schema = {
        "numeric_candidate_features": numeric_features,
        "state_categories_frozen_from_selection_training": state_categories,
        "all_anova_candidate_features": selection_audit["all_candidate_features"],
        "selected_features": selected_features,
        "rejected_columns": rejected_features,
        "input_dtypes": input_info["dtypes"],
        "target": DROUGHT,
        "selection_training_years": f"<= {TRAIN_END_YEAR}",
        "validation_years": list(VALIDATION_YEARS),
        "backtest_years": list(BACKTEST_YEARS),
        "deployment_year_range": [observed_year_min, observed_year_max],
        "clipping": [0.0, 100.0],
        "aggregation": (
            "county-area-weighted state-week values followed by an equal mean "
            "across weekly map dates within each state-year"
        ),
    }

    del data
    del deployment_model
    gc.collect()

    return {
        "history": history,
        "metrics": metrics,
        "schema": schema,
        "factors": factors,
        "input_information": input_info,
        "model_information": {
            "model_file": "drought_anova_ridge.joblib",
            "alpha": 1.0,
            "number_selected": len(selected_features),
            "selection_rows": selection_audit["selection_rows"],
        },
    }


# -----------------------------------------------------------------------------
# Compliance: validation-selected direct Tweedie or hurdle CatBoost
# -----------------------------------------------------------------------------


def compliance_feature_forbidden(column: str) -> bool:
    lower = column.lower()
    exact = {
        "pwsid",
        "pws_name",
        "system_name",
        "year",
        "state_fips",
        COMPLIANCE,
        COMPLIANCE_COUNT,
        COMPLIANCE_POPULATION,
        COMPLIANCE_RESIDUAL_FLAG,
    }
    if column in exact:
        return True
    if lower.startswith("target_"):
        return True
    if lower.startswith("context_only_") or "2026q2" in lower:
        return True
    if lower.endswith("_id"):
        return True
    if lower in {"county_fips", "county_list", "served_county_list"}:
        return True
    if "violation" in lower and not has_history_marker(lower):
        return True
    return False


def prepare_compliance_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str], list[str], int]:
    required = ["year", "state_fips", "state_abbreviation", COMPLIANCE]
    require_columns(data, required, "Compliance CSV")

    data = data.copy()
    data["year"] = pd.to_numeric(data["year"], errors="coerce").astype("Int64")
    data[COMPLIANCE] = pd.to_numeric(data[COMPLIANCE], errors="coerce")
    data["state_abbreviation"] = standardise_string(data["state_abbreviation"])
    data["state_fips"] = standardise_fips(data["state_fips"])
    if "master_record_type" in data.columns:
        data["master_record_type"] = standardise_string(data["master_record_type"])

    data.dropna(subset=["year", COMPLIANCE], inplace=True)
    data["year"] = data["year"].astype(int)
    if (data[COMPLIANCE] < 0).any():
        raise ValueError("Compliance target contains negative values.")

    base_year = int(data["year"].min())
    data["year_index"] = (data["year"] - base_year).astype(np.int16)
    categorical = [
        column
        for column in COMPLIANCE_PREFERRED_CATEGORICAL_COLUMNS
        if column in data.columns
    ]
    numeric: list[str] = []
    rejected: list[str] = []
    for column in data.columns:
        if column in categorical:
            continue
        if compliance_feature_forbidden(column):
            rejected.append(column)
            continue
        if pd.api.types.is_numeric_dtype(data[column]):
            numeric.append(column)

    if "year_index" not in numeric:
        numeric.append("year_index")

    forbidden_selected = [
        column
        for column in numeric + categorical
        if compliance_feature_forbidden(column)
    ]
    if forbidden_selected:
        raise ValueError(
            "Compliance leakage audit failed: " + ", ".join(forbidden_selected)
        )

    return data, numeric, categorical, rejected, base_year


def compliance_is_residual(data: pd.DataFrame) -> pd.Series:
    if COMPLIANCE_RESIDUAL_FLAG in data.columns:
        flag = pd.to_numeric(
            data[COMPLIANCE_RESIDUAL_FLAG], errors="coerce"
        ).fillna(0)
        return flag.eq(1)
    if "master_record_type" in data.columns:
        return data["master_record_type"].astype(str).eq(
            "unallocated_state_year_residual"
        )
    return pd.Series(False, index=data.index)


def representative_compliance_sample(
    data: pd.DataFrame,
    maximum_rows: int,
    minimum_positive_rows: int,
) -> pd.DataFrame:
    if len(data) <= maximum_rows:
        sample = data.copy()
        sample["_sample_weight"] = 1.0
        return sample

    random_sample = data.sample(n=maximum_rows, random_state=RANDOM_SEED).copy()
    positive_in_sample = int((random_sample[COMPLIANCE] > 0).sum())
    total_positive = int((data[COMPLIANCE] > 0).sum())
    if positive_in_sample >= minimum_positive_rows or total_positive <= positive_in_sample:
        random_sample["_sample_weight"] = 1.0
        return random_sample

    positive_rows = min(
        total_positive,
        max(minimum_positive_rows, int(maximum_rows * 0.10)),
        maximum_rows // 2,
    )
    zero_rows = maximum_rows - positive_rows
    positive_data = data[data[COMPLIANCE] > 0]
    zero_data = data[data[COMPLIANCE] <= 0]
    positive_sample = positive_data.sample(
        n=positive_rows, random_state=RANDOM_SEED
    )
    zero_sample = zero_data.sample(
        n=min(zero_rows, len(zero_data)), random_state=RANDOM_SEED + 1
    )
    sample = pd.concat([positive_sample, zero_sample], axis=0).sample(
        frac=1.0, random_state=RANDOM_SEED + 2
    )
    sample = sample.copy()
    positive_weight = len(positive_data) / max(len(positive_sample), 1)
    zero_weight = len(zero_data) / max(len(zero_sample), 1)
    sample["_sample_weight"] = np.where(
        sample[COMPLIANCE].to_numpy(dtype=float) > 0,
        positive_weight,
        zero_weight,
    )
    return sample


def select_compliance_numeric_features(
    tuning_sample: pd.DataFrame,
    candidates: list[str],
    maximum_features: int,
) -> list[str]:
    if len(candidates) <= maximum_features:
        return candidates

    target = np.log1p(tuning_sample[COMPLIANCE].to_numpy(dtype=float))
    scores: dict[str, float] = {}
    for column in candidates:
        values = pd.to_numeric(tuning_sample[column], errors="coerce").to_numpy(
            dtype=float
        )
        valid = np.isfinite(values) & np.isfinite(target)
        if valid.sum() < 100 or float(np.nanstd(values[valid])) <= 0:
            scores[column] = 0.0
            continue
        correlation = np.corrcoef(values[valid], target[valid])[0, 1]
        scores[column] = abs(float(correlation)) if np.isfinite(correlation) else 0.0

    mandatory = [
        column
        for column in candidates
        if column in MANDATORY_COMPLIANCE_HISTORY_FEATURES
        or has_history_marker(column)
    ]
    ranked = sorted(candidates, key=lambda column: scores[column], reverse=True)
    return list(dict.fromkeys(mandatory + ranked))[:maximum_features]


def compliance_frame(
    data: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for column in numeric_features:
        columns[column] = pd.to_numeric(
            data[column], errors="coerce"
        ).astype(np.float32)
    for column in categorical_features:
        columns[column] = standardise_string(data[column])
    ordered = numeric_features + categorical_features
    return pd.DataFrame(columns, index=data.index)[ordered]


def compliance_pool(
    data: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    label: np.ndarray | None = None,
    weight: np.ndarray | None = None,
) -> Pool:
    features = numeric_features + categorical_features
    return Pool(
        data=compliance_frame(data, numeric_features, categorical_features)[features],
        label=label,
        weight=weight,
        cat_features=categorical_features,
    )


def tweedie_parameters(
    configuration: TweedieConfiguration,
    iterations: int,
    threads: int,
    quick: bool,
    early_stopping: bool,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "iterations": iterations,
        "depth": configuration.depth,
        "learning_rate": configuration.learning_rate,
        "loss_function": f"Tweedie:variance_power={configuration.variance_power}",
        "eval_metric": f"Tweedie:variance_power={configuration.variance_power}",
        "l2_leaf_reg": 8,
        "random_seed": RANDOM_SEED,
        "allow_writing_files": False,
        "verbose": False if quick else 100,
        "thread_count": threads,
        "random_strength": 0.5,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.90,
    }
    if early_stopping:
        parameters.update({"od_type": "Iter", "od_wait": 30 if quick else 120})
    return parameters


def hurdle_classifier_parameters(
    configuration: HurdleConfiguration,
    iterations: int,
    threads: int,
    quick: bool,
    early_stopping: bool,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "iterations": iterations,
        "depth": configuration.classifier_depth,
        "learning_rate": configuration.classifier_learning_rate,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "l2_leaf_reg": 8,
        "random_seed": RANDOM_SEED,
        "allow_writing_files": False,
        "verbose": False if quick else 100,
        "thread_count": threads,
        "random_strength": 0.5,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.90,
    }
    if early_stopping:
        parameters.update({"od_type": "Iter", "od_wait": 30 if quick else 100})
    return parameters


def hurdle_magnitude_parameters(
    configuration: HurdleConfiguration,
    iterations: int,
    threads: int,
    quick: bool,
    early_stopping: bool,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "iterations": iterations,
        "depth": configuration.magnitude_depth,
        "learning_rate": configuration.magnitude_learning_rate,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "l2_leaf_reg": 8,
        "random_seed": RANDOM_SEED,
        "allow_writing_files": False,
        "verbose": False if quick else 100,
        "thread_count": threads,
        "random_strength": 0.5,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.90,
    }
    if early_stopping:
        parameters.update({"od_type": "Iter", "od_wait": 30 if quick else 100})
    return parameters


def compliance_state_keys(data: pd.DataFrame) -> list[str]:
    keys = ["state_fips", "state_abbreviation", "year"]
    return [column for column in keys if column in data.columns]


def compliance_actual_state_totals(data: pd.DataFrame) -> pd.DataFrame:
    keys = compliance_state_keys(data)
    return (
        data.groupby(keys, as_index=False, dropna=False)[COMPLIANCE]
        .sum()
        .rename(columns={COMPLIANCE: "actual_compliance"})
    )


def residual_baseline(
    state_year_reference: pd.DataFrame,
    residual_rows: pd.DataFrame,
    fit_end_year: int,
) -> dict[str, Any]:
    keys_without_year = ["state_fips", "state_abbreviation"]
    state_years = state_year_reference.loc[
        state_year_reference["year"] <= fit_end_year,
        keys_without_year + ["year"],
    ].drop_duplicates()
    residual = residual_rows[residual_rows["year"] <= fit_end_year]
    residual_totals = (
        residual.groupby(keys_without_year + ["year"], as_index=False)[COMPLIANCE]
        .sum()
        .rename(columns={COMPLIANCE: "residual_actual"})
    )
    state_years = state_years.merge(
        residual_totals,
        on=keys_without_year + ["year"],
        how="left",
        validate="one_to_one",
    )
    state_years["residual_actual"] = state_years["residual_actual"].fillna(0.0)

    rows: list[dict[str, Any]] = []
    for keys, group in state_years.groupby(keys_without_year, sort=False):
        recent = group.sort_values("year").tail(3)
        rows.append(
            {
                "state_fips": keys[0],
                "state_abbreviation": keys[1],
                "residual_prediction": float(recent["residual_actual"].mean()),
                "years_used": [int(year) for year in recent["year"].tolist()],
            }
        )
    table = pd.DataFrame(rows)
    global_fallback = float(state_years["residual_actual"].mean())
    return {
        "table": table,
        "global_fallback": global_fallback,
        "fit_end_year": fit_end_year,
        "strategy": (
            "Synthetic residual rows are excluded from the PWS CatBoost model. "
            "Each predicted state-year receives the state's mean residual "
            "contribution over its latest three observed training years; unseen "
            "states use the training-period global mean."
        ),
    }


def add_residual_baseline(
    predicted_ordinary_state: pd.DataFrame,
    baseline: dict[str, Any],
) -> pd.DataFrame:
    keys = ["state_fips", "state_abbreviation"]
    table = predicted_ordinary_state.merge(
        baseline["table"][[*keys, "residual_prediction"]],
        on=keys,
        how="left",
        validate="many_to_one",
    )
    table["residual_prediction"] = table["residual_prediction"].fillna(
        baseline["global_fallback"]
    )
    table["prediction"] = (
        table["prediction_ordinary"] + table["residual_prediction"]
    ).clip(lower=0.0)
    return table


def predict_compliance_chunks(
    data: pd.DataFrame,
    predictor: Callable[[Pool], np.ndarray],
    numeric_features: list[str],
    categorical_features: list[str],
    prediction_cap: float,
    chunk_size: int,
    label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = compliance_state_keys(data)
    grouped_parts: list[pd.DataFrame] = []
    sum_absolute_error = 0.0
    sum_squared_error = 0.0
    sum_target = 0.0
    sum_target_squared = 0.0
    row_count = 0

    for start in range(0, len(data), chunk_size):
        stop = min(start + chunk_size, len(data))
        chunk = data.iloc[start:stop]
        pool = compliance_pool(
            chunk, numeric_features, categorical_features
        )
        prediction = validate_and_clip_predictions(
            predictor(pool), 0.0, prediction_cap, label
        )
        actual = chunk[COMPLIANCE].to_numpy(dtype=float)
        error = prediction - actual
        sum_absolute_error += float(np.abs(error).sum())
        sum_squared_error += float(np.square(error).sum())
        sum_target += float(actual.sum())
        sum_target_squared += float(np.square(actual).sum())
        row_count += len(actual)

        part = chunk[keys].copy()
        part["prediction_ordinary"] = prediction
        grouped_parts.append(
            part.groupby(keys, as_index=False, dropna=False)[
                "prediction_ordinary"
            ].sum()
        )
        del pool
        del part
        gc.collect()

    state = (
        pd.concat(grouped_parts, ignore_index=True)
        .groupby(keys, as_index=False, dropna=False)["prediction_ordinary"]
        .sum()
    )
    mean_target = sum_target / max(row_count, 1)
    total_sum_squares = sum_target_squared - row_count * mean_target**2
    system_r2 = (
        1.0 - sum_squared_error / total_sum_squares
        if total_sum_squares > 0
        else float("nan")
    )
    system_metrics = {
        "mae": sum_absolute_error / max(row_count, 1),
        "rmse": math.sqrt(sum_squared_error / max(row_count, 1)),
        "r2": system_r2,
        "rows": int(row_count),
        "scope": "ordinary public-water-system rows only",
    }
    return state, system_metrics




def complete_ordinary_state_predictions(
    full_actuals: pd.DataFrame,
    ordinary_state: pd.DataFrame,
) -> pd.DataFrame:
    keys = compliance_state_keys(full_actuals)
    complete = full_actuals[keys].merge(
        ordinary_state[keys + ["prediction_ordinary"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    complete["prediction_ordinary"] = (
        complete["prediction_ordinary"].fillna(0.0).clip(lower=0.0)
    )
    return complete


def merge_compliance_predictions_with_actuals(
    full_actuals: pd.DataFrame,
    predicted_state: pd.DataFrame,
) -> pd.DataFrame:
    keys = compliance_state_keys(full_actuals)
    merged = full_actuals.merge(
        predicted_state[keys + ["prediction"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    merged["prediction"] = merged["prediction"].fillna(0.0).clip(lower=0.0)
    return merged


def evaluate_tweedie_candidate(
    configuration: TweedieConfiguration,
    tuning_train: pd.DataFrame,
    early_stop_validation: pd.DataFrame,
    full_validation_ordinary: pd.DataFrame,
    full_validation_actuals: pd.DataFrame,
    baseline: dict[str, Any],
    numeric_features: list[str],
    categorical_features: list[str],
    prediction_cap: float,
    args: argparse.Namespace,
) -> ComplianceCandidateResult:
    print("Training compliance candidate:", configuration.name)
    maximum_iterations = 180 if args.quick else 1_500
    model = CatBoostRegressor(
        **tweedie_parameters(
            configuration,
            maximum_iterations,
            args.threads,
            args.quick,
            early_stopping=True,
        )
    )
    train_pool = compliance_pool(
        tuning_train,
        numeric_features,
        categorical_features,
        label=tuning_train[COMPLIANCE].to_numpy(dtype=float),
        weight=tuning_train["_sample_weight"].to_numpy(dtype=float),
    )
    validation_pool = compliance_pool(
        early_stop_validation,
        numeric_features,
        categorical_features,
        label=early_stop_validation[COMPLIANCE].to_numpy(dtype=float),
    )
    started = time.perf_counter()
    model.fit(train_pool, eval_set=validation_pool, use_best_model=True)
    training_seconds = time.perf_counter() - started
    iterations = best_iteration(model, maximum_iterations)

    started = time.perf_counter()
    ordinary_state, system_metrics = predict_compliance_chunks(
        full_validation_ordinary,
        model.predict,
        numeric_features,
        categorical_features,
        prediction_cap,
        args.prediction_chunk_size,
        configuration.name,
    )
    ordinary_state = complete_ordinary_state_predictions(
        full_validation_actuals, ordinary_state
    )
    state_with_residual = add_residual_baseline(ordinary_state, baseline)
    validation_state = merge_compliance_predictions_with_actuals(
        full_validation_actuals, state_with_residual
    )
    prediction_seconds = time.perf_counter() - started
    state_metrics = regression_metrics(
        validation_state["actual_compliance"].to_numpy(dtype=float),
        validation_state["prediction"].to_numpy(dtype=float),
    )

    del model
    del train_pool
    del validation_pool
    gc.collect()
    return ComplianceCandidateResult(
        candidate=configuration.name,
        family="direct_tweedie",
        configuration=asdict(configuration),
        iterations={"tweedie": iterations},
        validation_state_metrics=state_metrics,
        validation_system_metrics=system_metrics,
        classifier_diagnostics=None,
        training_seconds=training_seconds,
        prediction_seconds=prediction_seconds,
        state_predictions=validation_state.rename(
            columns={"prediction": "predicted_compliance"}
        ),
    )


def train_hurdle_classifier_for_selection(
    configuration: HurdleConfiguration,
    tuning_train: pd.DataFrame,
    early_stop_validation: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    args: argparse.Namespace,
) -> tuple[CatBoostClassifier, int, dict[str, Any]]:
    maximum_iterations = 160 if args.quick else 1_000
    train_label = (tuning_train[COMPLIANCE].to_numpy(dtype=float) > 0).astype(int)
    validation_label = (
        early_stop_validation[COMPLIANCE].to_numpy(dtype=float) > 0
    ).astype(int)
    if np.unique(train_label).size != 2 or np.unique(validation_label).size != 2:
        raise ValueError("Hurdle classifier requires both target classes in each split.")

    model = CatBoostClassifier(
        **hurdle_classifier_parameters(
            configuration,
            maximum_iterations,
            args.threads,
            args.quick,
            early_stopping=True,
        )
    )
    train_pool = compliance_pool(
        tuning_train,
        numeric_features,
        categorical_features,
        label=train_label,
        weight=tuning_train["_sample_weight"].to_numpy(dtype=float),
    )
    validation_pool = compliance_pool(
        early_stop_validation,
        numeric_features,
        categorical_features,
        label=validation_label,
    )
    model.fit(train_pool, eval_set=validation_pool, use_best_model=True)
    probability = model.predict_proba(validation_pool)[:, 1]
    diagnostics = {
        "auc": float(roc_auc_score(validation_label, probability)),
        "average_precision": float(
            average_precision_score(validation_label, probability)
        ),
        "positive_rate_train": float(train_label.mean()),
        "positive_rate_validation": float(validation_label.mean()),
    }
    del train_pool
    del validation_pool
    gc.collect()
    return model, best_iteration(model, maximum_iterations), diagnostics


def predict_classifier_probability_chunks(
    model: CatBoostClassifier,
    data: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    chunk_size: int,
) -> np.ndarray:
    probability = np.empty(len(data), dtype=float)
    for start in range(0, len(data), chunk_size):
        stop = min(start + chunk_size, len(data))
        pool = compliance_pool(
            data.iloc[start:stop], numeric_features, categorical_features
        )
        probability[start:stop] = model.predict_proba(pool)[:, 1]
        del pool
        gc.collect()
    if not np.isfinite(probability).all():
        raise ValueError("Compliance hurdle classifier produced nonfinite probabilities.")
    return np.clip(probability, 0.0, 1.0)


def evaluate_hurdle_candidates(
    classifier: CatBoostClassifier,
    classifier_iterations: int,
    classifier_diagnostics: dict[str, Any],
    tuning_train: pd.DataFrame,
    early_stop_validation: pd.DataFrame,
    full_validation_ordinary: pd.DataFrame,
    full_validation_actuals: pd.DataFrame,
    baseline: dict[str, Any],
    numeric_features: list[str],
    categorical_features: list[str],
    prediction_cap: float,
    args: argparse.Namespace,
) -> list[ComplianceCandidateResult]:
    positive_train = tuning_train[tuning_train[COMPLIANCE] > 0].copy()
    positive_validation = early_stop_validation[
        early_stop_validation[COMPLIANCE] > 0
    ].copy()
    if positive_train.empty or positive_validation.empty:
        raise ValueError("Hurdle magnitude tuning requires positive rows.")

    probability = predict_classifier_probability_chunks(
        classifier,
        full_validation_ordinary,
        numeric_features,
        categorical_features,
        args.prediction_chunk_size,
    )
    depths = [6] if args.quick else [5, 6, 7]
    results: list[ComplianceCandidateResult] = []

    for depth in depths:
        configuration = HurdleConfiguration(magnitude_depth=depth)
        print("Training compliance candidate:", configuration.name)
        maximum_iterations = 160 if args.quick else 1_000
        model = CatBoostRegressor(
            **hurdle_magnitude_parameters(
                configuration,
                maximum_iterations,
                args.threads,
                args.quick,
                early_stopping=True,
            )
        )
        train_pool = compliance_pool(
            positive_train,
            numeric_features,
            categorical_features,
            label=np.log1p(positive_train[COMPLIANCE].to_numpy(dtype=float)),
            weight=positive_train["_sample_weight"].to_numpy(dtype=float),
        )
        validation_pool = compliance_pool(
            positive_validation,
            numeric_features,
            categorical_features,
            label=np.log1p(
                positive_validation[COMPLIANCE].to_numpy(dtype=float)
            ),
        )
        started = time.perf_counter()
        model.fit(train_pool, eval_set=validation_pool, use_best_model=True)
        training_seconds = time.perf_counter() - started
        magnitude_iterations = best_iteration(model, maximum_iterations)

        started = time.perf_counter()
        magnitude = np.empty(len(full_validation_ordinary), dtype=float)
        maximum_log = math.log1p(prediction_cap)
        for start in range(0, len(full_validation_ordinary), args.prediction_chunk_size):
            stop = min(start + args.prediction_chunk_size, len(full_validation_ordinary))
            pool = compliance_pool(
                full_validation_ordinary.iloc[start:stop],
                numeric_features,
                categorical_features,
            )
            magnitude[start:stop] = np.expm1(
                np.clip(model.predict(pool), 0.0, maximum_log)
            )
            del pool
            gc.collect()

        system_prediction = validate_and_clip_predictions(
            probability * magnitude,
            0.0,
            prediction_cap,
            configuration.name,
        )
        keys = compliance_state_keys(full_validation_ordinary)
        ordinary_state = (
            full_validation_ordinary[keys]
            .assign(prediction_ordinary=system_prediction)
            .groupby(keys, as_index=False, dropna=False)["prediction_ordinary"]
            .sum()
        )
        ordinary_state = complete_ordinary_state_predictions(
            full_validation_actuals, ordinary_state
        )
        state_with_residual = add_residual_baseline(ordinary_state, baseline)
        validation_state = merge_compliance_predictions_with_actuals(
            full_validation_actuals, state_with_residual
        )
        prediction_seconds = time.perf_counter() - started
        state_metrics = regression_metrics(
            validation_state["actual_compliance"].to_numpy(dtype=float),
            validation_state["prediction"].to_numpy(dtype=float),
        )
        system_metrics = regression_metrics(
            full_validation_ordinary[COMPLIANCE].to_numpy(dtype=float),
            system_prediction,
        )
        system_metrics["scope"] = "ordinary public-water-system rows only"
        results.append(
            ComplianceCandidateResult(
                candidate=configuration.name,
                family="hurdle",
                configuration=asdict(configuration),
                iterations={
                    "classifier": classifier_iterations,
                    "magnitude": magnitude_iterations,
                },
                validation_state_metrics=state_metrics,
                validation_system_metrics=system_metrics,
                classifier_diagnostics=classifier_diagnostics,
                training_seconds=training_seconds,
                prediction_seconds=prediction_seconds,
                state_predictions=validation_state.rename(
                    columns={"prediction": "predicted_compliance"}
                ),
            )
        )
        del model
        del train_pool
        del validation_pool
        del magnitude
        del system_prediction
        gc.collect()
    return results


def compliance_candidate_rank(result: ComplianceCandidateResult) -> tuple[float, float]:
    r2 = float(result.validation_state_metrics["r2"])
    if not math.isfinite(r2):
        r2 = -math.inf
    rmse = float(result.validation_state_metrics["rmse"])
    return r2, -rmse


def fit_compliance_tweedie(
    data: pd.DataFrame,
    configuration: TweedieConfiguration,
    iterations: int,
    numeric_features: list[str],
    categorical_features: list[str],
    args: argparse.Namespace,
) -> CatBoostRegressor:
    pool = compliance_pool(
        data,
        numeric_features,
        categorical_features,
        label=data[COMPLIANCE].to_numpy(dtype=float),
    )
    model = CatBoostRegressor(
        **tweedie_parameters(
            configuration,
            iterations,
            args.threads,
            args.quick,
            early_stopping=False,
        )
    )
    model.fit(pool)
    del pool
    gc.collect()
    return model


def fit_compliance_hurdle(
    data: pd.DataFrame,
    configuration: HurdleConfiguration,
    iterations: dict[str, int],
    numeric_features: list[str],
    categorical_features: list[str],
    args: argparse.Namespace,
) -> tuple[CatBoostClassifier, CatBoostRegressor]:
    labels = (data[COMPLIANCE].to_numpy(dtype=float) > 0).astype(int)
    classifier_pool = compliance_pool(
        data,
        numeric_features,
        categorical_features,
        label=labels,
    )
    classifier = CatBoostClassifier(
        **hurdle_classifier_parameters(
            configuration,
            iterations["classifier"],
            args.threads,
            args.quick,
            early_stopping=False,
        )
    )
    classifier.fit(classifier_pool)

    positive = data[data[COMPLIANCE] > 0]
    magnitude_pool = compliance_pool(
        positive,
        numeric_features,
        categorical_features,
        label=np.log1p(positive[COMPLIANCE].to_numpy(dtype=float)),
    )
    magnitude = CatBoostRegressor(
        **hurdle_magnitude_parameters(
            configuration,
            iterations["magnitude"],
            args.threads,
            args.quick,
            early_stopping=False,
        )
    )
    magnitude.fit(magnitude_pool)
    del classifier_pool
    del magnitude_pool
    gc.collect()
    return classifier, magnitude


def predict_compliance_hurdle_chunks(
    classifier: CatBoostClassifier,
    magnitude: CatBoostRegressor,
    data: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    prediction_cap: float,
    chunk_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    maximum_log = math.log1p(prediction_cap)

    def predictor(pool: Pool) -> np.ndarray:
        probability = classifier.predict_proba(pool)[:, 1]
        positive_magnitude = np.expm1(
            np.clip(magnitude.predict(pool), 0.0, maximum_log)
        )
        return probability * positive_magnitude

    return predict_compliance_chunks(
        data,
        predictor,
        numeric_features,
        categorical_features,
        prediction_cap,
        chunk_size,
        "Compliance hurdle",
    )


def compliance_factors(
    models: Iterable[CatBoostClassifier | CatBoostRegressor],
    features: list[str],
) -> list[str]:
    arrays = [np.asarray(model.get_feature_importance(), dtype=float) for model in models]
    average = np.mean(arrays, axis=0)
    table = pd.DataFrame({"feature": features, "importance": average})
    ignored = ("state_abbreviation", "master_record_type", "missing_flag")
    for word in ignored:
        table = table[~table["feature"].str.startswith(word)]
    table.sort_values("importance", ascending=False, inplace=True)
    factors = table["feature"].head(5).tolist()
    return factors if factors else features[:5]


def save_and_verify_compliance_models(
    output: Path,
    family: str,
    models: dict[str, Any],
    sample: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    prediction_cap: float,
) -> dict[str, str]:
    if family == "direct_tweedie":
        path = output / "compliance_tweedie.cbm"
        model = models["tweedie"]
        pool = compliance_pool(sample, numeric_features, categorical_features)
        expected = validate_and_clip_predictions(
            model.predict(pool), 0.0, prediction_cap, "Compliance Tweedie reload sample"
        )
        model.save_model(str(path))
        loaded = CatBoostRegressor()
        loaded.load_model(str(path))
        actual = validate_and_clip_predictions(
            loaded.predict(pool), 0.0, prediction_cap, "Compliance Tweedie reloaded"
        )
        verify_arrays_match(expected, actual, "compliance_tweedie.cbm")
        return {"tweedie": path.name}

    classifier_path = output / "compliance_hurdle_classifier.cbm"
    magnitude_path = output / "compliance_hurdle_magnitude.cbm"
    classifier = models["classifier"]
    magnitude = models["magnitude"]
    pool = compliance_pool(sample, numeric_features, categorical_features)
    maximum_log = math.log1p(prediction_cap)
    expected = validate_and_clip_predictions(
        classifier.predict_proba(pool)[:, 1]
        * np.expm1(np.clip(magnitude.predict(pool), 0.0, maximum_log)),
        0.0,
        prediction_cap,
        "Compliance hurdle reload sample",
    )
    classifier.save_model(str(classifier_path))
    magnitude.save_model(str(magnitude_path))
    loaded_classifier = CatBoostClassifier()
    loaded_classifier.load_model(str(classifier_path))
    loaded_magnitude = CatBoostRegressor()
    loaded_magnitude.load_model(str(magnitude_path))
    actual = validate_and_clip_predictions(
        loaded_classifier.predict_proba(pool)[:, 1]
        * np.expm1(np.clip(loaded_magnitude.predict(pool), 0.0, maximum_log)),
        0.0,
        prediction_cap,
        "Compliance hurdle reloaded",
    )
    verify_arrays_match(expected, actual, "compliance hurdle CatBoost artifacts")
    return {
        "classifier": classifier_path.name,
        "magnitude": magnitude_path.name,
    }


def train_compliance_pipeline(
    input_path: Path,
    output: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    print()
    print("=" * 72)
    print("COMPLIANCE PIPELINE")
    print("=" * 72)
    print("Reading:", input_path)

    raw = pd.read_csv(input_path, low_memory=False)
    input_info = file_information(input_path, raw, "year")
    data, numeric_candidates, categorical_features, rejected, base_year = (
        prepare_compliance_data(raw)
    )
    del raw
    gc.collect()

    residual_mask = compliance_is_residual(data)
    residual_rows = data[residual_mask].copy()
    ordinary = data[~residual_mask].copy()
    if ordinary.empty:
        raise ValueError("No ordinary PWS rows remain after residual separation.")

    if "pwsid" in ordinary.columns and ordinary.duplicated(["pwsid", "year"]).any():
        raise ValueError("Ordinary compliance rows contain duplicate pwsid/year keys.")

    ordinary["state_year_system_count_feature"] = (
        ordinary.groupby(["state_abbreviation", "year"], sort=False)[COMPLIANCE]
        .transform("size")
        .astype(np.int32)
    )
    if "state_year_system_count_feature" not in numeric_candidates:
        numeric_candidates.append("state_year_system_count_feature")

    print("All compliance rows:", len(data))
    print("Ordinary PWS rows:", len(ordinary))
    print("Synthetic residual rows:", len(residual_rows))

    if args.quick:
        quick_ordinary = select_quick_rows_by_year(
            ordinary,
            "year",
            rows_per_year=12_000,
            extra_positive_target=COMPLIANCE,
            positive_rows_per_year=2_000,
        )
        quick_residual = residual_rows.copy()
        data = pd.concat([quick_ordinary, quick_residual], ignore_index=True)
        ordinary = data[~compliance_is_residual(data)].copy()
        residual_rows = data[compliance_is_residual(data)].copy()
        ordinary["state_year_system_count_feature"] = (
            ordinary.groupby(["state_abbreviation", "year"], sort=False)[COMPLIANCE]
            .transform("size")
            .astype(np.int32)
        )
        print("QUICK MODE: compliance rows reduced to", len(data))

    actual_history = compliance_actual_state_totals(data)
    state_year_reference = data[
        compliance_state_keys(data)
    ].drop_duplicates().copy()
    observed_year_min = int(data["year"].min())
    observed_year_max = int(data["year"].max())
    del data
    gc.collect()

    train = ordinary[ordinary["year"] <= TRAIN_END_YEAR]
    validation_ordinary = ordinary[ordinary["year"].isin(VALIDATION_YEARS)]

    if train.empty or validation_ordinary.empty:
        raise ValueError("Compliance training or validation split is empty.")

    tuning_limit = 50_000 if args.quick else args.compliance_tuning_rows
    early_stop_limit = 25_000 if args.quick else args.compliance_eval_rows
    tuning_train = representative_compliance_sample(
        train,
        maximum_rows=min(tuning_limit, len(train)),
        minimum_positive_rows=500 if args.quick else 3_000,
    )
    early_stop_validation = validation_ordinary.sample(
        n=min(early_stop_limit, len(validation_ordinary)),
        random_state=RANDOM_SEED + 10,
    ).copy()

    numeric_features = select_compliance_numeric_features(
        tuning_train,
        numeric_candidates,
        args.max_compliance_numeric_features,
    )
    features = numeric_features + categorical_features
    print("Compliance numeric candidates:", len(numeric_candidates))
    print("Selected numeric features:", len(numeric_features))
    print("Categorical features:", categorical_features)
    for column in MANDATORY_COMPLIANCE_HISTORY_FEATURES:
        print(column, "FOUND" if column in numeric_features else "NOT FOUND")

    prediction_cap = derive_prediction_cap(train[COMPLIANCE], minimum=0.01)
    validation_actuals = actual_history[
        actual_history["year"].isin(VALIDATION_YEARS)
    ].copy()
    selection_residual_baseline = residual_baseline(
        state_year_reference, residual_rows, TRAIN_END_YEAR
    )

    results: list[ComplianceCandidateResult] = []
    powers = [1.3, 1.5, 1.7] if args.quick else [1.1, 1.3, 1.5, 1.7, 1.9]
    initial_configurations = [
        TweedieConfiguration(power, depth=6, learning_rate=0.03)
        for power in powers
    ]
    tested_names: set[str] = set()
    for configuration in initial_configurations:
        result = evaluate_tweedie_candidate(
            configuration,
            tuning_train,
            early_stop_validation,
            validation_ordinary,
            validation_actuals,
            selection_residual_baseline,
            numeric_features,
            categorical_features,
            prediction_cap,
            args,
        )
        results.append(result)
        tested_names.add(configuration.name)
        print_metric_summary(
            f"{configuration.name} validation state-year",
            result.validation_state_metrics,
        )

    if not args.quick:
        best_initial = max(
            [result for result in results if result.family == "direct_tweedie"],
            key=compliance_candidate_rank,
        )
        best_power = float(best_initial.configuration["variance_power"])
        expanded = [
            TweedieConfiguration(best_power, depth, learning_rate)
            for depth in [5, 6, 7]
            for learning_rate in [0.02, 0.03, 0.05]
        ]
        for configuration in expanded:
            if configuration.name in tested_names:
                continue
            result = evaluate_tweedie_candidate(
                configuration,
                tuning_train,
                early_stop_validation,
                validation_ordinary,
                validation_actuals,
                selection_residual_baseline,
                numeric_features,
                categorical_features,
                prediction_cap,
                args,
            )
            results.append(result)
            tested_names.add(configuration.name)
            print_metric_summary(
                f"{configuration.name} validation state-year",
                result.validation_state_metrics,
            )

    base_hurdle = HurdleConfiguration()
    classifier, classifier_iterations, classifier_diagnostics = (
        train_hurdle_classifier_for_selection(
            base_hurdle,
            tuning_train,
            early_stop_validation,
            numeric_features,
            categorical_features,
            args,
        )
    )
    hurdle_results = evaluate_hurdle_candidates(
        classifier,
        classifier_iterations,
        classifier_diagnostics,
        tuning_train,
        early_stop_validation,
        validation_ordinary,
        validation_actuals,
        selection_residual_baseline,
        numeric_features,
        categorical_features,
        prediction_cap,
        args,
    )
    for result in hurdle_results:
        results.append(result)
        print_metric_summary(
            f"{result.candidate} validation state-year",
            result.validation_state_metrics,
        )
    del classifier
    gc.collect()

    winner = max(results, key=compliance_candidate_rank)
    print("Compliance validation winner:", winner.candidate)
    print_metric_summary(
        "Compliance winning validation state-year",
        winner.validation_state_metrics,
    )

    candidate_rows: list[dict[str, Any]] = []
    for result in sorted(results, key=compliance_candidate_rank, reverse=True):
        candidate_rows.append(
            {
                "candidate": result.candidate,
                "family": result.family,
                "configuration": json.dumps(result.configuration, sort_keys=True),
                "iterations": json.dumps(result.iterations, sort_keys=True),
                "validation_state_mae": result.validation_state_metrics["mae"],
                "validation_state_rmse": result.validation_state_metrics["rmse"],
                "validation_state_r2": result.validation_state_metrics["r2"],
                "validation_system_mae": result.validation_system_metrics["mae"],
                "validation_system_rmse": result.validation_system_metrics["rmse"],
                "validation_system_r2": result.validation_system_metrics["r2"],
                "training_seconds": result.training_seconds,
                "prediction_seconds": result.prediction_seconds,
            }
        )
    pd.DataFrame(candidate_rows).to_csv(
        output / "compliance_validation_candidate_metrics.csv", index=False
    )

    validation_state_signal = winner.state_predictions[
        [
            "state_fips",
            "state_abbreviation",
            "year",
            "actual_compliance",
            "predicted_compliance",
        ]
    ].copy()

    del train
    del validation_ordinary
    del tuning_train
    del early_stop_validation
    gc.collect()

    through_validation = ordinary[ordinary["year"] <= VALIDATION_END_YEAR]
    backtest_ordinary = ordinary[ordinary["year"].isin(BACKTEST_YEARS)]
    if through_validation.empty or backtest_ordinary.empty:
        raise ValueError("Compliance evaluation refit or backtest split is empty.")

    if winner.family == "direct_tweedie":
        winning_tweedie = TweedieConfiguration(**winner.configuration)
        evaluation_model = fit_compliance_tweedie(
            through_validation,
            winning_tweedie,
            winner.iterations["tweedie"],
            numeric_features,
            categorical_features,
            args,
        )
        ordinary_state, backtest_system_metrics = predict_compliance_chunks(
            backtest_ordinary,
            evaluation_model.predict,
            numeric_features,
            categorical_features,
            prediction_cap,
            args.prediction_chunk_size,
            "Compliance Tweedie backtest",
        )
        evaluation_models: dict[str, Any] = {"tweedie": evaluation_model}
        winner_label = "CatBoost direct Tweedie"
    else:
        winning_hurdle = HurdleConfiguration(**winner.configuration)
        evaluation_classifier, evaluation_magnitude = fit_compliance_hurdle(
            through_validation,
            winning_hurdle,
            winner.iterations,
            numeric_features,
            categorical_features,
            args,
        )
        ordinary_state, backtest_system_metrics = predict_compliance_hurdle_chunks(
            evaluation_classifier,
            evaluation_magnitude,
            backtest_ordinary,
            numeric_features,
            categorical_features,
            prediction_cap,
            args.prediction_chunk_size,
        )
        evaluation_models = {
            "classifier": evaluation_classifier,
            "magnitude": evaluation_magnitude,
        }
        winner_label = "CatBoost hurdle classifier plus conditional log regressor"

    backtest_baseline = residual_baseline(
        state_year_reference, residual_rows, VALIDATION_END_YEAR
    )
    backtest_actuals = actual_history[
        actual_history["year"].isin(BACKTEST_YEARS)
    ].copy()
    ordinary_state = complete_ordinary_state_predictions(
        backtest_actuals, ordinary_state
    )
    state_with_residual = add_residual_baseline(ordinary_state, backtest_baseline)
    backtest_state = merge_compliance_predictions_with_actuals(
        backtest_actuals, state_with_residual
    ).rename(columns={"prediction": "predicted_compliance"})
    backtest_state_metrics = regression_metrics(
        backtest_state["actual_compliance"].to_numpy(dtype=float),
        backtest_state["predicted_compliance"].to_numpy(dtype=float),
    )
    print_metric_summary(
        "Compliance untouched backtest state-year", backtest_state_metrics
    )

    signal = pd.concat(
        [
            validation_state_signal[
                ["state_fips", "state_abbreviation", "year", "predicted_compliance"]
            ],
            backtest_state[
                ["state_fips", "state_abbreviation", "year", "predicted_compliance"]
            ],
        ],
        ignore_index=True,
    )
    history = actual_history.merge(
        signal,
        on=["state_fips", "state_abbreviation", "year"],
        how="left",
        validate="one_to_one",
    )
    history.sort_values(["state_fips", "year"], inplace=True)
    history.to_csv(output / "compliance_state_history.csv", index=False)

    del evaluation_models
    if winner.family == "direct_tweedie":
        del evaluation_model
    else:
        del evaluation_classifier
        del evaluation_magnitude
    del through_validation
    del backtest_ordinary
    gc.collect()

    if winner.family == "direct_tweedie":
        deployment_model = fit_compliance_tweedie(
            ordinary,
            winning_tweedie,
            winner.iterations["tweedie"],
            numeric_features,
            categorical_features,
            args,
        )
        deployment_models = {"tweedie": deployment_model}
    else:
        deployment_classifier, deployment_magnitude = fit_compliance_hurdle(
            ordinary,
            winning_hurdle,
            winner.iterations,
            numeric_features,
            categorical_features,
            args,
        )
        deployment_models = {
            "classifier": deployment_classifier,
            "magnitude": deployment_magnitude,
        }

    verification_sample = ordinary.sample(
        n=min(64, len(ordinary)), random_state=RANDOM_SEED
    ).copy()
    model_files = save_and_verify_compliance_models(
        output,
        winner.family,
        deployment_models,
        verification_sample,
        numeric_features,
        categorical_features,
        prediction_cap,
    )
    factors = compliance_factors(deployment_models.values(), features)
    deployment_residual_baseline = residual_baseline(
        state_year_reference, residual_rows, observed_year_max
    )

    information = {
        "model_label": winner_label,
        "family": winner.family,
        "winning_candidate": winner.candidate,
        "configuration": winner.configuration,
        "iterations": winner.iterations,
        "model_files": model_files,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "prediction_cap_from_selection_training": prediction_cap,
        "target": COMPLIANCE,
        "residual_treatment": {
            "residual_row_count": int(len(residual_rows)),
            "selection": {
                "fit_end_year": selection_residual_baseline["fit_end_year"],
                "strategy": selection_residual_baseline["strategy"],
                "global_fallback": selection_residual_baseline["global_fallback"],
            },
            "backtest": {
                "fit_end_year": backtest_baseline["fit_end_year"],
                "strategy": backtest_baseline["strategy"],
                "global_fallback": backtest_baseline["global_fallback"],
            },
            "deployment": {
                "fit_end_year": deployment_residual_baseline["fit_end_year"],
                "strategy": deployment_residual_baseline["strategy"],
                "global_fallback": deployment_residual_baseline["global_fallback"],
                "state_values": deployment_residual_baseline["table"].to_dict(
                    orient="records"
                ),
            },
        },
        "classifier_diagnostics": winner.classifier_diagnostics,
        "deployment_year_range": [observed_year_min, observed_year_max],
    }
    write_json(output / "compliance_model_information.json", information)

    metrics = {
        "validation": {
            "state_year": winner.validation_state_metrics,
            "system_row": winner.validation_system_metrics,
            "classifier_diagnostics": winner.classifier_diagnostics,
            "winner": winner.candidate,
        },
        "backtest": {
            "state_year": backtest_state_metrics,
            "system_row": backtest_system_metrics,
        },
        "backtest_by_year": metrics_by_year(
            backtest_state, "actual_compliance", "predicted_compliance"
        ),
        "candidate_ranking": candidate_rows,
    }
    schema = {
        "numeric_candidate_features": numeric_candidates,
        "selected_numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "ordered_features": features,
        "rejected_columns": rejected,
        "input_dtypes": input_info["dtypes"],
        "target": COMPLIANCE,
        "year_index_base": base_year,
        "selection_training_years": f"<= {TRAIN_END_YEAR}",
        "validation_years": list(VALIDATION_YEARS),
        "backtest_years": list(BACKTEST_YEARS),
        "deployment_year_range": information["deployment_year_range"],
        "prediction_cap": prediction_cap,
        "residual_rows_are_not_model_features": True,
    }

    del ordinary
    del residual_rows
    del actual_history
    del state_year_reference
    del deployment_models
    gc.collect()

    return {
        "history": history,
        "metrics": metrics,
        "schema": schema,
        "factors": factors,
        "input_information": input_info,
        "model_information": information,
        "best_model_label": winner_label,
    }


# -----------------------------------------------------------------------------
# Final merged history, project metadata, and completion marker
# -----------------------------------------------------------------------------


def percentile_boundaries(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.empty:
        raise ValueError("No training values were available for normalization.")
    low = float(np.percentile(clean, 5))
    high = float(np.percentile(clean, 95))
    if high <= low:
        low = float(clean.min())
        high = float(clean.max())
    if high <= low:
        high = low + 1.0
    return {"low": low, "high": high}


def merge_histories(
    electricity: pd.DataFrame,
    drought: pd.DataFrame,
    compliance: pd.DataFrame,
) -> pd.DataFrame:
    frames = [electricity.copy(), drought.copy(), compliance.copy()]
    for frame in frames:
        frame["state_fips"] = standardise_fips(frame["state_fips"])
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype(int)

    history = frames[0].merge(
        frames[1],
        on=["state_fips", "year"],
        how="outer",
        validate="one_to_one",
        suffixes=("", "_drought"),
    )
    history = history.merge(
        frames[2],
        on=["state_fips", "year"],
        how="outer",
        validate="one_to_one",
        suffixes=("", "_compliance"),
    )

    abbreviation_columns = [
        column for column in history.columns if column.startswith("state_abbreviation")
    ]
    if abbreviation_columns:
        combined = history[abbreviation_columns[0]]
        for column in abbreviation_columns[1:]:
            combined = combined.combine_first(history[column])
        history["state_abbreviation"] = combined
        history.drop(
            columns=[
                column
                for column in abbreviation_columns
                if column != "state_abbreviation"
            ],
            inplace=True,
        )

    state_name_columns = [
        column for column in history.columns if column.startswith("state_name")
    ]
    if state_name_columns:
        combined_name = history[state_name_columns[0]]
        for column in state_name_columns[1:]:
            combined_name = combined_name.combine_first(history[column])
        history["state_name"] = combined_name
        history.drop(
            columns=[column for column in state_name_columns if column != "state_name"],
            inplace=True,
        )

    expected = [
        "state_fips",
        "state_name",
        "state_abbreviation",
        "year",
        "actual_saidi",
        "predicted_saidi",
        "actual_saifi",
        "predicted_saifi",
        "actual_drought",
        "predicted_drought",
        "actual_compliance",
        "predicted_compliance",
    ]
    for column in expected:
        if column not in history.columns:
            history[column] = np.nan
    history = history[expected].sort_values(["state_fips", "year"]).reset_index(
        drop=True
    )
    return history


def validate_history_integrity(history: pd.DataFrame) -> None:
    if history.duplicated(["state_fips", "year"]).any():
        raise ValueError("state_model_history.csv would contain duplicate state-year rows.")

    prediction_columns = [
        "predicted_saidi",
        "predicted_saifi",
        "predicted_drought",
        "predicted_compliance",
    ]
    for column in prediction_columns:
        early_nonnull = history.loc[history["year"] <= TRAIN_END_YEAR, column].notna()
        if early_nonnull.any():
            raise ValueError(
                f"{column} contains pre-2019 values. Earlier years must remain null "
                "unless they are generated by a separate leakage-free fold."
            )
        if history[column].dropna().empty:
            raise ValueError(f"{column} contains no chronological model signals.")
        if not np.isfinite(history[column].dropna().to_numpy(dtype=float)).all():
            raise ValueError(f"{column} contains nonfinite values.")


def build_project_outputs(
    output: Path,
    electricity: dict[str, Any],
    drought: dict[str, Any],
    compliance: dict[str, Any],
    args: argparse.Namespace,
    started_at: datetime,
) -> None:
    history = merge_histories(
        electricity["history"], drought["history"], compliance["history"]
    )
    validate_history_integrity(history)
    history.to_csv(output / "state_model_history.csv", index=False)

    normalization_training = history[history["year"] <= TRAIN_END_YEAR]
    boundaries = {
        "saidi": percentile_boundaries(normalization_training["actual_saidi"]),
        "saifi": percentile_boundaries(normalization_training["actual_saifi"]),
        "compliance": percentile_boundaries(
            normalization_training["actual_compliance"]
        ),
    }

    evaluation_metrics = {
        "metrics_are_final": not args.quick,
        "quick_mode_warning": (
            "Quick-mode metrics are non-final smoke-test results."
            if args.quick
            else None
        ),
        "electricity": electricity["metrics"],
        "drought": drought["metrics"],
        "compliance": compliance["metrics"],
    }
    write_json(output / "evaluation_metrics.json", evaluation_metrics)

    feature_schema = {
        "electricity": electricity["schema"],
        "drought": drought["schema"],
        "compliance": compliance["schema"],
    }
    write_json(output / "feature_schema.json", feature_schema)

    latest_observed_year = int(history["year"].max())
    metadata = {
        "latest_observed_year": latest_observed_year,
        "normalization_boundaries": boundaries,
        "projection": {
            "method": (
                "recursive one-year-ahead baseline scenario using the recent "
                "chronological model-signal trend, trend damping, and gradual "
                "movement toward each state's historical conditions"
            ),
            "recent_years_used": 5,
            "damping": 0.50,
            "historical_typical_value_weight": 0.20,
            "drought_change_damping": 0.50,
            "drought_historical_mean_weight": 0.20,
            "historical_signal_policy": (
                "2019-2020 signals are from models trained through 2018; "
                "2021-2024 signals are from frozen models refitted through 2020; "
                "earlier predictions are intentionally null."
            ),
        },
        "backtest_rmse": {
            "saidi": electricity["metrics"]["saidi"]["backtest"]["state_year"][
                "rmse"
            ],
            "saifi": electricity["metrics"]["saifi"]["backtest"]["state_year"][
                "rmse"
            ],
            "drought": drought["metrics"]["backtest"]["state_year"]["rmse"],
            "compliance": compliance["metrics"]["backtest"]["state_year"][
                "rmse"
            ],
        },
        "best_models": {
            **MODEL_LABELS,
            "compliance": compliance["best_model_label"],
        },
        "main_historical_model_inputs": {
            "saidi": electricity["factors"]["saidi"],
            "saifi": electricity["factors"]["saifi"],
            "drought": drought["factors"],
            "compliance": compliance["factors"],
        },
        "important_warning": (
            "Years after the observed data are damped baseline scenario "
            "projections, not direct forecasts with known future weather, "
            "population, demand, or infrastructure. The displayed ranges use "
            "backtest RMSE and are approximate rather than formally calibrated "
            "prediction intervals."
        ),
        "quick_mode": args.quick,
        "metrics_are_final": not args.quick,
        "random_seed": RANDOM_SEED,
        "software_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "catboost": catboost.__version__,
            "joblib": joblib.__version__,
        },
        "input_files": {
            "electricity": electricity["input_information"],
            "drought": drought["input_information"],
            "compliance": compliance["input_information"],
        },
        "chronological_evaluation": {
            "selection_training": f"years <= {TRAIN_END_YEAR}",
            "validation_selection": list(VALIDATION_YEARS),
            "evaluation_refit": f"years <= {VALIDATION_END_YEAR}",
            "untouched_backtest": list(BACKTEST_YEARS),
            "deployment_refit": "all observed years after evaluation",
        },
        "selected_model_information": {
            "electricity": electricity["model_information"],
            "drought": drought["model_information"],
            "compliance": compliance["model_information"],
        },
        "evaluation_metrics_file": "evaluation_metrics.json",
        "feature_schema_file": "feature_schema.json",
        "run_started_utc": started_at.isoformat(),
        "run_completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "project_metadata.json", metadata)

    required_files = [
        "electricity_state_history.csv",
        "drought_state_history.csv",
        "compliance_state_history.csv",
        "state_model_history.csv",
        "project_metadata.json",
        "evaluation_metrics.json",
        "feature_schema.json",
        "electricity_model_information.json",
        "drought_anova_ridge.joblib",
        "compliance_model_information.json",
        "saidi_high_event_classifier.cbm",
        "saidi_normal_regressor.cbm",
        "saidi_high_regressor.cbm",
        "saifi_log_catboost.cbm",
    ]
    for name in required_files:
        if not (output / name).exists():
            raise FileNotFoundError(f"Required output was not created: {name}")

    compliance_files = compliance["model_information"]["model_files"].values()
    for name in compliance_files:
        if not (output / name).exists():
            raise FileNotFoundError(f"Required compliance artifact was not created: {name}")

    completion = {
        "status": "complete",
        "quick_mode": args.quick,
        "metrics_are_final": not args.quick,
        "latest_observed_year": latest_observed_year,
        "saved_model_folder": str(MODEL_FOLDER),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "required_file_count": len(list(output.iterdir())) + 1,
    }
    # This is deliberately the final file written inside the temporary folder.
    write_json(output / "training_complete.json", completion)


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_arguments()
    configure_threads(args.threads)
    started_at = datetime.now(timezone.utc)
    temporary = prepare_output_folder(args.force)

    input_paths = {
        "electricity": Path(args.electricity_file),
        "drought": Path(args.drought_file),
        "compliance": Path(args.compliance_file),
    }
    for label, path in input_paths.items():
        if not path.exists():
            cleanup_temporary_folder(temporary)
            raise FileNotFoundError(f"Required {label} CSV was not found: {path}")

    try:
        if args.quick:
            print("QUICK MODE: all metrics produced by this run are non-final.")

        electricity = train_electricity_pipeline(
            input_paths["electricity"], temporary, args
        )
        drought = train_drought_pipeline(input_paths["drought"], temporary, args)
        compliance = train_compliance_pipeline(
            input_paths["compliance"], temporary, args
        )
        build_project_outputs(
            temporary, electricity, drought, compliance, args, started_at
        )
        publish_output_folder(temporary, args.force)
    except Exception:
        cleanup_temporary_folder(temporary)
        raise

    print()
    print("=" * 72)
    print("ALL FOUR PIPELINES WERE TRAINED, VERIFIED, AND SAVED")
    print("=" * 72)
    print("Saved models:", MODEL_FOLDER)
    if args.quick:
        print("This was a non-final quick-mode smoke test.")
    print("Run the projector with:")
    print("py -3.13 stress_score_cli.py")


if __name__ == "__main__":
    main()
