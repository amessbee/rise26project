import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from catboost import CatBoostClassifier
from catboost import Pool

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


FILE_PATH = r"D:\RISE_Project\Feature_CSVs\electricity_features_final_sequence_2013_2024.csv"

SAIDI = "target_saidi_minutes_per_customer"
SAIFI = "target_saifi_interruptions_per_customer"
CUSTOMERS = "sample_weight_reporting_customers"


# These columns change from month to month.
MONTHLY_COLUMNS = [
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


# CatBoost treats these as labels, not continuous numbers.
CATEGORY_COLUMNS = [
    "utility_number",
    "state_fips",
    "ownership",
    "reporting_standard",
]


def main():
    electricity = pd.read_csv(FILE_PATH)
    annual = make_one_row_per_sequence(electricity)
    annual, features = prepare_features(annual)

    print("Original monthly rows:", len(electricity))
    print("Annual utility-state sequences:", len(annual))
    print("Model features:", len(features))

    train = annual[annual["target_year"] <= 2018].copy()
    validation = annual[annual["target_year"].between(2019, 2020)].copy()
    backtest = annual[annual["target_year"] >= 2021].copy()

    # This order is used by CatBoost's time-aware categorical processing.
    train = train.sort_values(["target_year", "sequence_id"]).reset_index(drop=True)
    validation = validation.sort_values(
        ["target_year", "sequence_id"]
    ).reset_index(drop=True)
    backtest = backtest.sort_values(
        ["target_year", "sequence_id"]
    ).reset_index(drop=True)

    run_saidi(train, validation, backtest, features)
    run_saifi(train, validation, backtest, features)


def make_one_row_per_sequence(electricity):
    # Every annual target must have exactly 12 input-month rows.
    rows_per_sequence = electricity.groupby("sequence_id").size()
    months_per_sequence = electricity.groupby("sequence_id")["month"].nunique()

    if not rows_per_sequence.eq(12).all():
        raise ValueError("Some annual sequences do not have exactly 12 rows.")

    if not months_per_sequence.eq(12).all():
        raise ValueError("Some annual sequences do not have 12 unique months.")

    if electricity.duplicated(["sequence_id", "month"]).any():
        raise ValueError("A sequence contains a duplicated month.")

    # These annual values are repeated on all 12 monthly rows.
    annual_columns = [
        "sequence_id",
        "utility_number",
        "state_fips",
        "state_name",
        "target_year",
        "ownership",
        "reporting_standard",
        SAIDI,
        SAIFI,
        CUSTOMERS,
    ]

    for column in electricity.columns:
        if column.startswith("previous_year_"):
            annual_columns.append(column)
        elif column.startswith("previous_3_year_"):
            annual_columns.append(column)

    annual_columns = list(dict.fromkeys(annual_columns))

    # Confirm that repeated annual values really are constant within a sequence.
    for column in annual_columns:
        if column == "sequence_id":
            continue

        different_values = electricity.groupby("sequence_id")[column].nunique(
            dropna=False
        )

        if different_values.gt(1).any():
            raise ValueError(column + " changes inside an annual sequence.")

    annual = electricity[annual_columns].drop_duplicates("sequence_id").copy()

    # Turn the 12 monthly rows into January, February, ... December columns.
    monthly = electricity.pivot(
        index="sequence_id",
        columns="month",
        values=MONTHLY_COLUMNS,
    )

    monthly.columns = [
        column + "_month_" + str(month).zfill(2)
        for column, month in monthly.columns
    ]

    monthly = monthly.reset_index()
    annual = annual.merge(monthly, on="sequence_id", how="left", validate="one_to_one")
    return annual


def prepare_features(annual):
    # Fill only categorical missing values. CatBoost handles numerical NaN values itself.
    for column in CATEGORY_COLUMNS:
        annual[column] = annual[column].fillna("Unknown").astype(str)

    not_features = [
        "sequence_id",
        "state_name",
        "target_year",
        SAIDI,
        SAIFI,
        CUSTOMERS,
    ]

    features = []

    for column in annual.columns:
        if column not in not_features:
            features.append(column)

    return annual, features


def get_weights(data):
    weights = data[CUSTOMERS].astype(float)
    return (weights / weights.mean()).to_numpy()


def make_pool(data, features, target=None, use_log=False):
    labels = None

    if target is not None:
        labels = data[target].to_numpy()

        if use_log:
            labels = np.log1p(labels)

        return Pool(
            data=data[features],
            label=labels,
            cat_features=CATEGORY_COLUMNS,
            weight=get_weights(data),
        )

    return Pool(
        data=data[features],
        cat_features=CATEGORY_COLUMNS,
    )


def train_regressor(train, validation, features, target, use_log, depth=7):
    model = CatBoostRegressor(
        iterations=2000,
        depth=depth,
        learning_rate=0.03,
        loss_function="RMSE",
        eval_metric="RMSE",
        l2_leaf_reg=7,
        random_seed=67,
        has_time=True,
        allow_writing_files=False,
        verbose=100,
    )

    training_pool = make_pool(train, features, target, use_log)
    validation_pool = make_pool(validation, features, target, use_log)

    model.fit(
        training_pool,
        eval_set=validation_pool,
        early_stopping_rounds=150,
        use_best_model=True,
    )

    return model


def predict_regressor(model, data, features, use_log):
    prediction_pool = make_pool(data, features)
    predictions = model.predict(prediction_pool)

    if use_log:
        predictions = np.expm1(predictions)

    return np.clip(predictions, 0, None)


def train_high_saidi_models(train, validation, features):
    # The high-SAIDI boundary is learned from training data only.
    threshold = train[SAIDI].quantile(0.90)

    train = train.copy()
    validation = validation.copy()

    train["high_saidi"] = (train[SAIDI] > threshold).astype(int)
    validation["high_saidi"] = (validation[SAIDI] > threshold).astype(int)

    classifier = CatBoostClassifier(
        iterations=1500,
        depth=6,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="Logloss",
        l2_leaf_reg=7,
        random_seed=67,
        has_time=True,
        allow_writing_files=False,
        verbose=100,
    )

    classifier.fit(
        make_pool(train, features, "high_saidi"),
        eval_set=make_pool(validation, features, "high_saidi"),
        early_stopping_rounds=150,
        use_best_model=True,
    )

    normal_train = train[train["high_saidi"] == 0].copy()
    normal_validation = validation[validation["high_saidi"] == 0].copy()

    high_train = train[train["high_saidi"] == 1].copy()
    high_validation = validation[validation["high_saidi"] == 1].copy()

    normal_model = train_regressor(
        normal_train,
        normal_validation,
        features,
        SAIDI,
        use_log=True,
        depth=7,
    )

    high_model = train_regressor(
        high_train,
        high_validation,
        features,
        SAIDI,
        use_log=True,
        depth=5,
    )

    return classifier, normal_model, high_model, threshold


def predict_high_saidi_models(classifier, normal_model, high_model, data, features):
    prediction_pool = make_pool(data, features)

    high_probability = classifier.predict_proba(prediction_pool)[:, 1]
    normal_predictions = np.expm1(normal_model.predict(prediction_pool))
    high_predictions = np.expm1(high_model.predict(prediction_pool))

    predictions = (
        (1 - high_probability) * normal_predictions
        + high_probability * high_predictions
    )

    return np.clip(predictions, 0, None)


def run_saidi(train, validation, backtest, features):
    train = train.dropna(subset=[SAIDI]).copy()
    validation = validation.dropna(subset=[SAIDI]).copy()
    backtest = backtest.dropna(subset=[SAIDI]).copy()

    print("\n================ SAIDI ================")
    print("Training rows:", len(train))
    print("Validation rows:", len(validation))
    print("Backtest rows:", len(backtest))

    # Candidate 1: one CatBoost model trained on log1p(SAIDI).
    direct_model = train_regressor(
        train,
        validation,
        features,
        SAIDI,
        use_log=True,
    )

    direct_validation = predict_regressor(
        direct_model, validation, features, use_log=True
    )

    direct_backtest = predict_regressor(
        direct_model, backtest, features, use_log=True
    )

    # Candidate 2: classify high-SAIDI years, then blend normal/high regressors.
    classifier, normal_model, high_model, threshold = train_high_saidi_models(
        train, validation, features
    )

    two_stage_validation = predict_high_saidi_models(
        classifier, normal_model, high_model, validation, features
    )

    two_stage_backtest = predict_high_saidi_models(
        classifier, normal_model, high_model, backtest, features
    )

    print("\nHigh-SAIDI training threshold:", round(threshold, 3), "minutes")
    direct_rmse = print_results(
        "DIRECT LOG-CATBOOST VALIDATION", validation, direct_validation, SAIDI
    )
    two_stage_rmse = print_results(
        "TWO-STAGE CATBOOST VALIDATION",
        validation,
        two_stage_validation,
        SAIDI,
    )

    if two_stage_rmse < direct_rmse:
        chosen_name = "Two-stage CatBoost"
        chosen_backtest = two_stage_backtest
    else:
        chosen_name = "Direct log-CatBoost"
        chosen_backtest = direct_backtest

    print("\nChosen SAIDI model from validation:", chosen_name)
    print_results("FINAL SAIDI BACKTEST", backtest, chosen_backtest, SAIDI)

    compare_with_previous_year(
        backtest,
        chosen_backtest,
        SAIDI,
        "previous_year_saidi_minutes_per_customer",
    )


def run_saifi(train, validation, backtest, features):
    train = train.dropna(subset=[SAIFI]).copy()
    validation = validation.dropna(subset=[SAIFI]).copy()
    backtest = backtest.dropna(subset=[SAIFI]).copy()

    print("\n================ SAIFI ================")
    print("Training rows:", len(train))
    print("Validation rows:", len(validation))
    print("Backtest rows:", len(backtest))

    # Try raw SAIFI and log1p(SAIFI). Choose using validation only.
    raw_model = train_regressor(
        train,
        validation,
        features,
        SAIFI,
        use_log=False,
    )

    log_model = train_regressor(
        train,
        validation,
        features,
        SAIFI,
        use_log=True,
    )

    raw_validation = predict_regressor(
        raw_model, validation, features, use_log=False
    )
    log_validation = predict_regressor(
        log_model, validation, features, use_log=True
    )

    raw_backtest = predict_regressor(
        raw_model, backtest, features, use_log=False
    )
    log_backtest = predict_regressor(
        log_model, backtest, features, use_log=True
    )

    raw_rmse = print_results(
        "RAW CATBOOST VALIDATION", validation, raw_validation, SAIFI
    )
    log_rmse = print_results(
        "LOG CATBOOST VALIDATION", validation, log_validation, SAIFI
    )

    if log_rmse < raw_rmse:
        chosen_name = "Log-CatBoost"
        chosen_backtest = log_backtest
    else:
        chosen_name = "Raw CatBoost"
        chosen_backtest = raw_backtest

    print("\nChosen SAIFI model from validation:", chosen_name)
    print_results("FINAL SAIFI BACKTEST", backtest, chosen_backtest, SAIFI)

    compare_with_previous_year(
        backtest,
        chosen_backtest,
        SAIFI,
        "previous_year_saifi_interruptions_per_customer",
    )


def make_state_year_results(data, predictions, target):
    results = data[["state_fips", "target_year", target, CUSTOMERS]].copy()
    results["prediction"] = predictions
    results["actual_times_customers"] = results[target] * results[CUSTOMERS]
    results["prediction_times_customers"] = (
        results["prediction"] * results[CUSTOMERS]
    )

    state_year = results.groupby(["state_fips", "target_year"])[
        ["actual_times_customers", "prediction_times_customers", CUSTOMERS]
    ].sum()

    state_year["actual"] = (
        state_year["actual_times_customers"] / state_year[CUSTOMERS]
    )

    state_year["prediction"] = (
        state_year["prediction_times_customers"] / state_year[CUSTOMERS]
    )

    return state_year


def print_results(name, data, predictions, target):
    weights = data[CUSTOMERS]

    utility_mae = mean_absolute_error(
        data[target], predictions, sample_weight=weights
    )
    utility_rmse = np.sqrt(
        mean_squared_error(data[target], predictions, sample_weight=weights)
    )
    utility_r2 = r2_score(data[target], predictions, sample_weight=weights)

    state_year = make_state_year_results(data, predictions, target)

    state_mae = mean_absolute_error(
        state_year["actual"], state_year["prediction"]
    )
    state_rmse = np.sqrt(
        mean_squared_error(state_year["actual"], state_year["prediction"])
    )
    state_r2 = r2_score(state_year["actual"], state_year["prediction"])

    print("\n" + name)
    print("Customer-weighted utility MAE:", round(utility_mae, 3))
    print("Customer-weighted utility RMSE:", round(utility_rmse, 3))
    print("Customer-weighted utility R2:", round(utility_r2, 3))
    print("State-year MAE:", round(state_mae, 3))
    print("State-year RMSE:", round(state_rmse, 3))
    print("State-year R2:", round(state_r2, 3))

    return state_rmse


def compare_with_previous_year(data, model_predictions, target, previous_column):
    available = data[previous_column].notna().to_numpy()

    if available.sum() == 0:
        print("No previous-year baseline values are available.")
        return

    covered_data = data.loc[available].copy()
    covered_model_predictions = model_predictions[available]
    baseline_predictions = covered_data[previous_column].to_numpy()

    print("\nPERSISTENCE BASELINE COMPARISON")
    print("Rows with a previous-year baseline:", len(covered_data))

    print_results(
        "PREVIOUS-YEAR BASELINE",
        covered_data,
        baseline_predictions,
        target,
    )

    print_results(
        "CHOSEN CATBOOST ON THE SAME ROWS",
        covered_data,
        covered_model_predictions,
        target,
    )


if __name__ == "__main__":
    main()
