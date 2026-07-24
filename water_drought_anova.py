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


FILE_PATH = r"D:\RISE_Project\Feature_CSVs\water_drought_features_county_week_2010_2024.csv"
TARGET = "target_drought_severity_0_100"
WEIGHT = "county_land_area_weight_within_state"


def main():
    drought = pd.read_csv(FILE_PATH)
    drought["map_date"] = pd.to_datetime(drought["map_date"])
    drought, features = add_state_columns(drought)

    train = drought[drought["year"] <= 2018].dropna(subset=[TARGET]).copy()
    validation = drought[
        drought["year"].between(2019, 2020)
    ].dropna(subset=[TARGET]).copy()
    backtest = drought[drought["year"] >= 2021].dropna(subset=[TARGET]).copy()

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()

    X_train = scaler.fit_transform(imputer.fit_transform(train[features]))
    X_validation = scaler.transform(imputer.transform(validation[features]))
    X_backtest = scaler.transform(imputer.transform(backtest[features]))

    number_selected = min(20, len(features))
    selector = SelectKBest(score_func=f_regression, k=number_selected)

    X_train = selector.fit_transform(X_train, train[TARGET])
    X_validation = selector.transform(X_validation)
    X_backtest = selector.transform(X_backtest)

    model = Ridge(alpha=1.0)
    model.fit(X_train, train[TARGET], sample_weight=get_weights(train))

    validation_predictions = np.clip(model.predict(X_validation), 0, 100)
    backtest_predictions = np.clip(model.predict(X_backtest), 0, 100)

    print_results("VALIDATION", validation, validation_predictions)
    print_results("BACKTEST", backtest, backtest_predictions)

    selected_names = np.array(features)[selector.get_support()]
    selected_scores = selector.scores_[selector.get_support()]
    selected_p_values = selector.pvalues_[selector.get_support()]

    selected = pd.DataFrame(
        {
            "feature": selected_names,
            "f_score": selected_scores,
            "p_value": selected_p_values,
        }
    ).sort_values("f_score", ascending=False)

    print("\nSELECTED FEATURES")
    print(selected.to_string(index=False))

    return model


def add_state_columns(drought):
    state_columns = pd.get_dummies(
        drought["state_abbreviation"].fillna("Unknown"),
        prefix="state",
        dtype=float,
    )
    drought = pd.concat([drought, state_columns], axis=1)

    excluded = [
        "county_fips",
        "state_fips",
        "year",
        TARGET,
        WEIGHT,
        "usdm_cumulative_order_valid",
    ]

    features = []
    for column in drought.select_dtypes(include="number").columns:
        if column not in excluded:
            features.append(column)

    return drought, features


def get_weights(data):
    weights = data[WEIGHT].fillna(0).clip(lower=0)
    if weights.mean() == 0:
        return np.ones(len(data))
    return (weights / weights.mean()).to_numpy()


def print_results(name, data, predictions):
    county_mae = mean_absolute_error(data[TARGET], predictions)
    county_rmse = np.sqrt(mean_squared_error(data[TARGET], predictions))
    county_r2 = r2_score(data[TARGET], predictions)

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
    print("County-week MAE:", round(county_mae, 3))
    print("County-week RMSE:", round(county_rmse, 3))
    print("County-week R2:", round(county_r2, 3))
    print("State-year MAE:", round(state_mae, 3))
    print("State-year RMSE:", round(state_rmse, 3))
    print("State-year R2:", round(state_r2, 3))


if __name__ == "__main__":
    main()
