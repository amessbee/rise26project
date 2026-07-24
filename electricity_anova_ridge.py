import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

FILE_PATH = (
    r"D:\RISE_Project\Feature_CSVs"
    r"\electricity_features_final_sequence_2013_2024.csv"
)

SAIDI_TARGET = "target_saidi_minutes_per_customer"

SAIFI_TARGET = "target_saifi_interruptions_per_customer"

WEIGHT_COLUMN = "sample_weight_reporting_customers"


def main():
    electricity = pd.read_csv(FILE_PATH)

    annual, features = make_annual_table(electricity)

    print(
        "Annual sequences:",
        len(annual),
    )

    print(
        "Available features:",
        len(features),
    )

    print("\nSAIDI F-TEST + RIDGE")

    saidi_model = train_pipeline(
        annual,
        features,
        SAIDI_TARGET,
    )

    print("\nSAIFI F-TEST + RIDGE")

    saifi_model = train_pipeline(
        annual,
        features,
        SAIFI_TARGET,
    )

    return saidi_model, saifi_model


def make_annual_table(electricity):
    not_features = [
        "sequence_id",
        "utility_number",
        "state_fips",
        "input_year",
        "target_year",
        "month",
        SAIDI_TARGET,
        SAIFI_TARGET,
        WEIGHT_COLUMN,
    ]

    number_columns = electricity.select_dtypes(include="number").columns

    numerical_features = []

    for column in number_columns:
        if column not in not_features:
            numerical_features.append(column)

    # Calculate annual summaries from the 12 months.
    summaries = electricity.groupby("sequence_id")[numerical_features].agg(
        [
            "mean",
            "min",
            "max",
            "std",
        ]
    )

    # Pandas uses min and max rather than
    # minimum and maximum.

    new_names = []

    for feature_name, calculation in summaries.columns:
        new_name = feature_name + "_" + calculation

        new_names.append(new_name)

    summaries.columns = new_names

    information_columns = [
        "utility_number",
        "state_fips",
        "state_name",
        "ownership",
        "reporting_standard",
        "target_year",
        SAIDI_TARGET,
        SAIFI_TARGET,
        WEIGHT_COLUMN,
    ]

    information = electricity.groupby("sequence_id")[information_columns].first()

    annual = information.join(summaries).reset_index()

    category_data = annual[
        [
            "state_name",
            "ownership",
            "reporting_standard",
        ]
    ].fillna("Unknown")

    category_columns = pd.get_dummies(
        category_data,
        dtype=float,
    )

    annual = pd.concat(
        [
            annual,
            category_columns,
        ],
        axis=1,
    )

    features = []

    for column in summaries.columns:
        features.append(column)

    for column in category_columns.columns:
        features.append(column)

    return annual, features


def get_weights(data):
    weights = pd.to_numeric(
        data[WEIGHT_COLUMN],
        errors="coerce",
    )

    median_weight = weights.median()

    if pd.isna(median_weight):
        median_weight = 1

    weights = weights.fillna(median_weight)

    weights = weights.clip(lower=1)

    weights = weights / weights.mean()

    return weights.to_numpy()


def prepare_target(values, target):
    values = values.to_numpy(dtype=float)

    if target == SAIDI_TARGET:
        values = np.log1p(values)

    return values


def convert_predictions(
    predictions,
    training_data,
    target,
):
    if target == SAIDI_TARGET:
        largest_log = np.log1p(training_data[target]).max()

        predictions = np.clip(
            predictions,
            0,
            largest_log,
        )

        predictions = np.expm1(predictions)

    else:
        predictions = np.clip(
            predictions,
            0,
            None,
        )

    return predictions


def calculate_state_metrics(
    data,
    predictions,
    target,
):
    results = data[
        [
            "state_name",
            "target_year",
            target,
            WEIGHT_COLUMN,
        ]
    ].copy()

    results["prediction"] = predictions

    results["weight"] = pd.to_numeric(
        results[WEIGHT_COLUMN],
        errors="coerce",
    ).fillna(0)

    results = results[results["weight"] > 0].copy()

    results["actual_weighted"] = results[target] * results["weight"]

    results["prediction_weighted"] = results["prediction"] * results["weight"]

    state_results = results.groupby(
        [
            "state_name",
            "target_year",
        ]
    )[
        [
            "actual_weighted",
            "prediction_weighted",
            "weight",
        ]
    ].sum()

    actual = state_results["actual_weighted"] / state_results["weight"]

    predicted = state_results["prediction_weighted"] / state_results["weight"]

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    r2 = r2_score(
        actual,
        predicted,
    )

    return mae, rmse, r2


def train_pipeline(
    annual,
    features,
    target,
):
    data = annual.dropna(subset=[target]).copy()

    train = data[data["target_year"] <= 2018]

    validation = data[
        data["target_year"].between(
            2019,
            2020,
        )
    ]

    backtest = data[data["target_year"] >= 2021]

    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )

    scaler = StandardScaler()

    X_train = imputer.fit_transform(train[features])

    X_validation = imputer.transform(validation[features])

    X_train = scaler.fit_transform(X_train)

    X_validation = scaler.transform(X_validation)

    y_train = prepare_target(
        train[target],
        target,
    )

    training_weights = get_weights(train)

    k_options = [
        20,
        50,
        100,
    ]

    alpha_options = [
        0.1,
        1.0,
        10.0,
    ]

    best_rmse = float("inf")
    best_k = None
    best_alpha = None

    for k in k_options:
        if k > len(features):
            continue

        selector = SelectKBest(
            score_func=f_regression,
            k=k,
        )

        selected_train = selector.fit_transform(
            X_train,
            y_train,
        )

        selected_validation = selector.transform(X_validation)

        for alpha in alpha_options:
            model = Ridge(alpha=alpha)

            model.fit(
                selected_train,
                y_train,
                sample_weight=training_weights,
            )

            predictions = model.predict(selected_validation)

            predictions = convert_predictions(
                predictions,
                train,
                target,
            )

            mae, rmse, r2 = calculate_state_metrics(
                validation,
                predictions,
                target,
            )

            print()
            print("Selected features:", k)
            print("Ridge alpha:", alpha)
            print(
                "Validation state MAE:",
                round(mae, 3),
            )
            print(
                "Validation state RMSE:",
                round(rmse, 3),
            )
            print(
                "Validation state R2:",
                round(r2, 3),
            )

            if rmse < best_rmse:
                best_rmse = rmse
                best_k = k
                best_alpha = alpha

    print()
    print("BEST SETTINGS")
    print(
        "Selected features:",
        best_k,
    )
    print(
        "Ridge alpha:",
        best_alpha,
    )

    final_training = pd.concat(
        [
            train,
            validation,
        ]
    )

    final_imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )

    final_scaler = StandardScaler()

    X_final = final_imputer.fit_transform(final_training[features])

    X_backtest = final_imputer.transform(backtest[features])

    X_final = final_scaler.fit_transform(X_final)

    X_backtest = final_scaler.transform(X_backtest)

    y_final = prepare_target(
        final_training[target],
        target,
    )

    final_selector = SelectKBest(
        score_func=f_regression,
        k=best_k,
    )

    X_final = final_selector.fit_transform(
        X_final,
        y_final,
    )

    X_backtest = final_selector.transform(X_backtest)

    final_model = Ridge(alpha=best_alpha)

    final_model.fit(
        X_final,
        y_final,
        sample_weight=get_weights(final_training),
    )

    predictions = final_model.predict(X_backtest)

    predictions = convert_predictions(
        predictions,
        final_training,
        target,
    )

    utility_mae = mean_absolute_error(
        backtest[target],
        predictions,
    )

    utility_rmse = np.sqrt(
        mean_squared_error(
            backtest[target],
            predictions,
        )
    )

    utility_r2 = r2_score(
        backtest[target],
        predictions,
    )

    state_mae, state_rmse, state_r2 = calculate_state_metrics(
        backtest,
        predictions,
        target,
    )

    print()
    print("FINAL BACKTEST RESULTS")
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

    selected_feature_names = np.array(features)[final_selector.get_support()]

    selected_f_scores = final_selector.scores_[final_selector.get_support()]

    selected_p_values = final_selector.pvalues_[final_selector.get_support()]

    selected_table = pd.DataFrame(
        {
            "feature": selected_feature_names,
            "f_score": selected_f_scores,
            "p_value": selected_p_values,
        }
    )

    selected_table = selected_table.sort_values(
        "f_score",
        ascending=False,
    )

    print()
    print("TOP SELECTED FEATURES")
    print(selected_table.head(20))

    return final_model


if __name__ == "__main__":
    main()
