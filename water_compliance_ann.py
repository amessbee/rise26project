import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


FILE_PATH = r"D:\RISE_Project\Feature_CSVs\water_compliance_features_public_water_system_year_2010_2024.csv"
TARGET = "target_health_based_violation_rate_contribution_per_100000_state_residents"


def main():
    compliance = pd.read_csv(FILE_PATH)
    compliance, features = add_category_columns(compliance)

    train = compliance[compliance["year"] <= 2018].dropna(subset=[TARGET]).copy()
    validation = compliance[
        compliance["year"].between(2019, 2020)
    ].dropna(subset=[TARGET]).copy()
    backtest = compliance[compliance["year"] >= 2021].dropna(subset=[TARGET]).copy()

    print("Training rows:", len(train))
    print("Validation rows:", len(validation))
    print("Backtest rows:", len(backtest))
    print("Features:", len(features))

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()

    X_train = scaler.fit_transform(imputer.fit_transform(train[features]))
    X_validation = scaler.transform(imputer.transform(validation[features]))
    X_backtest = scaler.transform(imputer.transform(backtest[features]))

    y_train = np.log1p(train[TARGET])

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        random_state=67,
    )

    model.fit(X_train, y_train)

    largest_log = y_train.max()
    validation_log = np.clip(model.predict(X_validation), 0, largest_log)
    backtest_log = np.clip(model.predict(X_backtest), 0, largest_log)

    validation_predictions = np.expm1(validation_log)
    backtest_predictions = np.expm1(backtest_log)

    print_results("VALIDATION", validation, validation_predictions)
    print_results("BACKTEST", backtest, backtest_predictions)
    return model


def add_category_columns(compliance):
    categories = compliance[
        ["state_abbreviation", "master_record_type"]
    ].fillna("Unknown")

    category_columns = pd.get_dummies(categories, dtype=float)
    compliance = pd.concat([compliance, category_columns], axis=1)

    excluded = [
        "state_fips",
        "year",
        TARGET,
        "target_health_based_violation_count",
        "state_population_persons_for_target",
        "context_only_2026q2_population_served_count",
        "context_only_2026q2_service_connections_count",
    ]

    features = []
    for column in compliance.select_dtypes(include="number").columns:
        if column not in excluded:
            features.append(column)

    return compliance, features


def print_results(name, data, predictions):
    system_mae = mean_absolute_error(data[TARGET], predictions)
    system_rmse = np.sqrt(mean_squared_error(data[TARGET], predictions))
    system_r2 = r2_score(data[TARGET], predictions)

    results = data[["state_fips", "year", TARGET]].copy()
    results["prediction"] = predictions
    state_year = results.groupby(["state_fips", "year"])[
        [TARGET, "prediction"]
    ].sum()

    state_mae = mean_absolute_error(state_year[TARGET], state_year["prediction"])
    state_rmse = np.sqrt(
        mean_squared_error(state_year[TARGET], state_year["prediction"])
    )
    state_r2 = r2_score(state_year[TARGET], state_year["prediction"])

    print("\n" + name + " RESULTS")
    print("System-year MAE:", round(system_mae, 6))
    print("System-year RMSE:", round(system_rmse, 6))
    print("System-year R2:", round(system_r2, 3))
    print("State-year MAE:", round(state_mae, 3))
    print("State-year RMSE:", round(state_rmse, 3))
    print("State-year R2:", round(state_r2, 3))


if __name__ == "__main__":
    main()
