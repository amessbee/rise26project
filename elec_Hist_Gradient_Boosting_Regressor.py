import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# --------------------------------------------------
# 1. READ ELECTRICITY DATA
# --------------------------------------------------

electricity = pd.read_csv(
    r"D:\RISE_Project\Feature_CSVs"
    r"\electricity_features_final_sequence_2013_2024.csv"
)


# --------------------------------------------------
# 2. COLUMN NAMES
# --------------------------------------------------

SAIDI_TARGET = "target_saidi_minutes_per_customer"

SAIFI_TARGET = "target_saifi_interruptions_per_customer"

WEIGHT_COLUMN = "sample_weight_reporting_customers"


# These columns change from month to month.
monthly_features = [
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


# These are categories rather than measurements.
category_features = [
    "state_name",
    "ownership",
    "reporting_standard",
]


# These describe each annual sequence.
information_columns = [
    "utility_number",
    "utility_name",
    "state_fips",
    "state_name",
    "target_year",
    "input_year",
    "ownership",
    "reporting_standard",
    SAIDI_TARGET,
    SAIFI_TARGET,
    WEIGHT_COLUMN,
]


# --------------------------------------------------
# 3. CREATE ONE ROW PER ANNUAL SEQUENCE
# --------------------------------------------------


def make_annual_table():
    # Keep identifying information and targets.
    information = electricity.groupby("sequence_id")[information_columns].first()

    # Put each month into separate columns.
    monthly_table = electricity.pivot_table(
        index="sequence_id",
        columns="month",
        values=monthly_features,
        aggfunc="first",
    )

    # Rename columns into understandable names,
    # such as total_sales_month_1.
    new_monthly_names = []

    for feature_name, month_number in monthly_table.columns:
        new_name = feature_name + "_month_" + str(int(month_number))

        new_monthly_names.append(new_name)

    monthly_table.columns = new_monthly_names

    # Find the remaining annual numerical features.
    number_columns = electricity.select_dtypes(include="number").columns

    columns_not_used_as_annual_features = (
        information_columns
        + monthly_features
        + [
            "month",
            "sequence_id",
        ]
    )

    annual_features = []

    for column in number_columns:
        if column not in columns_not_used_as_annual_features:
            annual_features.append(column)

    # Annual features repeat across the 12 months,
    # so keep only one copy per sequence.
    annual_history = electricity.groupby("sequence_id")[annual_features].first()

    # Join everything into one annual table.
    annual = information.join(annual_history)

    annual = annual.join(monthly_table)

    annual = annual.reset_index()

    # Tell the model that these columns are categories.
    for column in category_features:
        annual[column] = annual[column].fillna("Unknown").astype("category")

    # Final list of model inputs.
    model_features = []

    for column in annual_features:
        model_features.append(column)

    for column in new_monthly_names:
        model_features.append(column)

    for column in category_features:
        model_features.append(column)

    return annual, model_features


# --------------------------------------------------
# 4. CHRONOLOGICAL SPLIT
# --------------------------------------------------


def split_data(annual, target):
    # Remove sequences without the required target.
    data = annual.dropna(subset=[target]).copy()

    train = data[data["target_year"] <= 2018]

    validation = data[
        data["target_year"].between(
            2019,
            2020,
        )
    ]

    backtest = data[data["target_year"] >= 2021]

    return train, validation, backtest


# --------------------------------------------------
# 5. CUSTOMER WEIGHTS
# --------------------------------------------------


def get_training_weights(data):
    weights = data[WEIGHT_COLUMN].copy()

    median_weight = weights.median()

    if pd.isna(median_weight):
        median_weight = 1

    weights = weights.fillna(median_weight)

    weights = weights.clip(lower=1)

    # Keep weights from becoming unnecessarily huge.
    weights = weights / weights.mean()

    return weights


# --------------------------------------------------
# 6. MAKE THE MODEL
# --------------------------------------------------


def make_model(
    loss,
    learning_rate,
    max_leaf_nodes,
):
    model = HistGradientBoostingRegressor(
        loss=loss,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        max_iter=300,
        min_samples_leaf=20,
        l2_regularization=1.0,
        categorical_features="from_dtype",
        early_stopping=True,
        random_state=67,
    )

    return model


# --------------------------------------------------
# 7. PREPARE TARGET FOR TRAINING
# --------------------------------------------------


def prepare_target(data, target):
    # SAIDI has extreme values, so use log1p.
    if target == SAIDI_TARGET:
        prepared_target = np.log1p(data[target])

    # SAIFI does not need the same transformation.
    else:
        prepared_target = data[target]

    return prepared_target


# --------------------------------------------------
# 8. MAKE PREDICTIONS
# --------------------------------------------------


def make_predictions(
    model,
    training_data,
    prediction_data,
    model_features,
    target,
):
    predictions = model.predict(prediction_data[model_features])

    # Convert logged SAIDI back into minutes.
    if target == SAIDI_TARGET:
        largest_training_log = np.log1p(training_data[target]).max()

        predictions = np.clip(
            predictions,
            0,
            largest_training_log,
        )

        predictions = np.expm1(predictions)

    # SAIFI predictions cannot be negative.
    else:
        predictions = np.clip(
            predictions,
            0,
            None,
        )

    return predictions


# --------------------------------------------------
# 9. CALCULATE UTILITY AND STATE METRICS
# --------------------------------------------------


def calculate_metrics(
    data,
    predictions,
    target,
):
    # Utility-level results.
    actual_utility = data[target]

    utility_mae = mean_absolute_error(
        actual_utility,
        predictions,
    )

    utility_rmse = np.sqrt(
        mean_squared_error(
            actual_utility,
            predictions,
        )
    )

    utility_r2 = r2_score(
        actual_utility,
        predictions,
    )

    # Prepare state-level customer-weighted results.
    results = data[
        [
            "state_name",
            "target_year",
            target,
            WEIGHT_COLUMN,
        ]
    ].copy()

    results["prediction"] = predictions

    results["weight"] = results[WEIGHT_COLUMN].fillna(0)

    # Only use utilities with a known customer count
    # in the customer-weighted state calculation.
    results = results[results["weight"] > 0].copy()

    results["actual_times_weight"] = results[target] * results["weight"]

    results["prediction_times_weight"] = results["prediction"] * results["weight"]

    state_results = results.groupby(
        [
            "state_name",
            "target_year",
        ]
    )[
        [
            "actual_times_weight",
            "prediction_times_weight",
            "weight",
        ]
    ].sum()

    state_results["actual"] = (
        state_results["actual_times_weight"] / state_results["weight"]
    )

    state_results["prediction"] = (
        state_results["prediction_times_weight"] / state_results["weight"]
    )

    actual_state = state_results["actual"]
    predicted_state = state_results["prediction"]

    state_mae = mean_absolute_error(
        actual_state,
        predicted_state,
    )

    state_rmse = np.sqrt(
        mean_squared_error(
            actual_state,
            predicted_state,
        )
    )

    state_r2 = r2_score(
        actual_state,
        predicted_state,
    )

    return (
        utility_mae,
        utility_rmse,
        utility_r2,
        state_mae,
        state_rmse,
        state_r2,
    )


# --------------------------------------------------
# 10. DISPLAY MODEL RESULTS
# --------------------------------------------------


def display_results(
    name,
    data,
    predictions,
    target,
):
    (
        utility_mae,
        utility_rmse,
        utility_r2,
        state_mae,
        state_rmse,
        state_r2,
    ) = calculate_metrics(
        data,
        predictions,
        target,
    )

    print()
    print(name)

    print(
        "Utility MAE:",
        round(utility_mae, 3),
    )

    print(
        "Utility RMSE:",
        round(utility_rmse, 3),
    )

    print(
        "Utility R2:",
        round(utility_r2, 3),
    )

    print(
        "State MAE:",
        round(state_mae, 3),
    )

    print(
        "State RMSE:",
        round(state_rmse, 3),
    )

    print(
        "State R2:",
        round(state_r2, 3),
    )


# --------------------------------------------------
# 11. BASELINE MODELS
# --------------------------------------------------


def display_baselines(
    train,
    backtest,
    target,
):
    print()
    print("BACKTEST BASELINES")

    # Customer-weighted training mean.
    train_weights = get_training_weights(train)

    training_mean = np.average(
        train[target],
        weights=train_weights,
    )

    mean_predictions = np.full(
        len(backtest),
        training_mean,
    )

    display_results(
        "Training mean baseline",
        backtest,
        mean_predictions,
        target,
    )

    # Previous-year persistence baseline.
    if target == SAIDI_TARGET:
        previous_year_column = "previous_year_saidi_minutes_per_customer"

    else:
        previous_year_column = "previous_year_saifi_interruptions_per_customer"

    persistence_data = backtest.dropna(subset=[previous_year_column]).copy()

    persistence_predictions = persistence_data[previous_year_column].to_numpy()

    display_results(
        "Previous-year persistence baseline",
        persistence_data,
        persistence_predictions,
        target,
    )


# --------------------------------------------------
# 12. TRAIN ONE PIPELINE
# --------------------------------------------------


def train_pipeline(
    annual,
    model_features,
    target,
):
    train, validation, backtest = split_data(
        annual,
        target,
    )

    print()
    print("Training sequences:", len(train))
    print(
        "Validation sequences:",
        len(validation),
    )
    print(
        "Backtest sequences:",
        len(backtest),
    )

    # Try different tree settings.
    learning_rate_options = [
        0.03,
        0.08,
    ]

    leaf_options = [
        15,
        31,
        63,
    ]

    # Losses appropriate for each target.
    if target == SAIDI_TARGET:
        loss_options = [
            "squared_error",
            "absolute_error",
        ]

    else:
        loss_options = [
            "squared_error",
            "absolute_error",
            "poisson",
        ]

    best_state_rmse = float("inf")
    best_loss = None
    best_learning_rate = None
    best_leaf_count = None

    training_weights = get_training_weights(train)

    training_target = prepare_target(
        train,
        target,
    )

    for loss in loss_options:
        for learning_rate in learning_rate_options:
            for leaf_count in leaf_options:

                model = make_model(
                    loss,
                    learning_rate,
                    leaf_count,
                )

                model.fit(
                    train[model_features],
                    training_target,
                    sample_weight=training_weights,
                )

                validation_predictions = make_predictions(
                    model,
                    train,
                    validation,
                    model_features,
                    target,
                )

                (
                    utility_mae,
                    utility_rmse,
                    utility_r2,
                    state_mae,
                    state_rmse,
                    state_r2,
                ) = calculate_metrics(
                    validation,
                    validation_predictions,
                    target,
                )

                print()
                print("Loss:", loss)
                print(
                    "Learning rate:",
                    learning_rate,
                )
                print(
                    "Maximum leaf nodes:",
                    leaf_count,
                )
                print(
                    "Validation state MAE:",
                    round(state_mae, 3),
                )
                print(
                    "Validation state RMSE:",
                    round(state_rmse, 3),
                )
                print(
                    "Validation state R2:",
                    round(state_r2, 3),
                )

                # Select using state-level RMSE because
                # the application produces state results.
                if state_rmse < best_state_rmse:
                    best_state_rmse = state_rmse
                    best_loss = loss
                    best_learning_rate = learning_rate
                    best_leaf_count = leaf_count

    print()
    print("BEST SETTINGS")
    print("Loss:", best_loss)
    print(
        "Learning rate:",
        best_learning_rate,
    )
    print(
        "Maximum leaf nodes:",
        best_leaf_count,
    )
    print(
        "Best validation state RMSE:",
        round(best_state_rmse, 3),
    )

    # Combine train and validation after choosing settings.
    final_training_data = pd.concat(
        [
            train,
            validation,
        ]
    )

    final_training_weights = get_training_weights(final_training_data)

    final_training_target = prepare_target(
        final_training_data,
        target,
    )

    final_model = make_model(
        best_loss,
        best_learning_rate,
        best_leaf_count,
    )

    final_model.fit(
        final_training_data[model_features],
        final_training_target,
        sample_weight=final_training_weights,
    )

    backtest_predictions = make_predictions(
        final_model,
        final_training_data,
        backtest,
        model_features,
        target,
    )

    print()
    print("FINAL GRADIENT-BOOSTING RESULTS")

    display_results(
        "Gradient-boosting backtest",
        backtest,
        backtest_predictions,
        target,
    )

    display_baselines(
        train,
        backtest,
        target,
    )

    return final_model


# --------------------------------------------------
# 13. RUN BOTH ELECTRICITY MODELS
# --------------------------------------------------


def main():
    annual, model_features = make_annual_table()

    print(
        "Original monthly rows:",
        len(electricity),
    )

    print(
        "Annual sequences:",
        len(annual),
    )

    print(
        "Model features:",
        len(model_features),
    )

    print()
    print("SAIDI GRADIENT-BOOSTING MODEL")

    saidi_model = train_pipeline(
        annual,
        model_features,
        SAIDI_TARGET,
    )

    print()
    print("SAIFI GRADIENT-BOOSTING MODEL")

    saifi_model = train_pipeline(
        annual,
        model_features,
        SAIFI_TARGET,
    )

    return saidi_model, saifi_model


if __name__ == "__main__":
    main()
