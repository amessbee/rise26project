import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


FILE_PATH = r"D:\RISE_Project\Feature_CSVs\water_drought_features_county_week_2010_2024.csv"
TARGET = "target_drought_severity_0_100"
WEIGHT = "county_land_area_weight_within_state"


def main():
    drought = pd.read_csv(FILE_PATH)
    drought["map_date"] = pd.to_datetime(drought["map_date"])

    drought, features = add_state_columns(drought)
    train, validation, backtest = split_data(drought)

    print("Training rows:", len(train))
    print("Validation rows:", len(validation))
    print("Backtest rows:", len(backtest))
    print("Features:", len(features))

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()

    X_train = imputer.fit_transform(train[features])
    X_validation = imputer.transform(validation[features])
    X_backtest = imputer.transform(backtest[features])

    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)
    X_backtest = scaler.transform(X_backtest)

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        random_state=67,
    )

    model.fit(
        X_train,
        train[TARGET],
        sample_weight=get_weights(train),
    )

    validation_predictions = np.clip(model.predict(X_validation), 0, 100)
    backtest_predictions = np.clip(model.predict(X_backtest), 0, 100)

    print_results("VALIDATION", validation, validation_predictions)
    print_results("BACKTEST", backtest, backtest_predictions)

    return model


def add_state_columns(drought):
    state_columns = pd.get_dummies(
        drought["state_abbreviation"].fillna("Unknown"),
        prefix="state",
        dtype=float,
    )

    drought = pd.concat([drought, state_columns], axis=1)

    not_features = [
        "county_fips",
        "state_fips",
        "map_date",
        "year",
        TARGET,
        WEIGHT,
        "usdm_cumulative_order_valid",
    ]

    features = []

    for column in drought.select_dtypes(include="number").columns:
        if column not in not_features:
            features.append(column)

    return drought, features


def split_data(drought):
    drought = drought.dropna(subset=[TARGET]).copy()
    train = drought[drought["year"] <= 2018].copy()
    validation = drought[drought["year"].between(2019, 2020)].copy()
    backtest = drought[drought["year"] >= 2021].copy()
    return train, validation, backtest


def get_weights(data):
    weights = data[WEIGHT].fillna(0).clip(lower=0)
    if weights.mean() == 0:
        return np.ones(len(data))
    return (weights / weights.mean()).to_numpy()


def print_results(name, data, predictions):
    row_mae = mean_absolute_error(data[TARGET], predictions)
    row_rmse = np.sqrt(mean_squared_error(data[TARGET], predictions))
    row_r2 = r2_score(data[TARGET], predictions)

    results = data[["state_fips", "year", "map_date", TARGET, WEIGHT]].copy()
    results["prediction"] = predictions
    results["weight"] = results[WEIGHT].fillna(0)
    results["actual_weighted"] = results[TARGET] * results["weight"]
    results["prediction_weighted"] = results["prediction"] * results["weight"]

    state_week = results.groupby(
        ["state_fips", "year", "map_date"]
    )[["actual_weighted", "prediction_weighted", "weight"]].sum()

    state_week = state_week[state_week["weight"] > 0].copy()
    state_week["actual"] = state_week["actual_weighted"] / state_week["weight"]
    state_week["prediction"] = state_week["prediction_weighted"] / state_week["weight"]

    state_year = state_week.groupby(["state_fips", "year"])[
        ["actual", "prediction"]
    ].mean()

    state_mae = mean_absolute_error(state_year["actual"], state_year["prediction"])
    state_rmse = np.sqrt(
        mean_squared_error(state_year["actual"], state_year["prediction"])
    )
    state_r2 = r2_score(state_year["actual"], state_year["prediction"])

    print("\n" + name + " RESULTS")
    print("County-week MAE:", round(row_mae, 3))
    print("County-week RMSE:", round(row_rmse, 3))
    print("County-week R2:", round(row_r2, 3))
    print("State-year MAE:", round(state_mae, 3))
    print("State-year RMSE:", round(state_rmse, 3))
    print("State-year R2:", round(state_r2, 3))


if __name__ == "__main__":
    main()
