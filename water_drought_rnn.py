import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from tensorflow import keras
from tensorflow.keras import layers


FILE_PATH = r"D:\RISE_Project\Feature_CSVs\water_drought_features_county_week_2010_2024.csv"
TARGET = "target_drought_severity_0_100"
WEIGHT = "county_land_area_weight_within_state"
SEQUENCE_LENGTH = 8


def main():
    drought = pd.read_csv(FILE_PATH)
    drought["map_date"] = pd.to_datetime(drought["map_date"])
    drought = drought.sort_values(["county_fips", "map_date"]).reset_index(drop=True)

    features = get_features(drought)
    X_rows = preprocess_rows(drought, features)
    X, information = make_sequences(drought, X_rows)

    print("Sequence model: RNN")
    print("Sequences:", len(information))
    print("Weeks per sequence:", SEQUENCE_LENGTH)
    print("Features per week:", len(features))

    model = train_model(X, information)
    return model


def get_features(drought):
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
    return features


def preprocess_rows(drought, features):
    training_rows = drought["year"] <= 2018
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()

    imputer.fit(drought.loc[training_rows, features])
    training_values = imputer.transform(drought.loc[training_rows, features])
    scaler.fit(training_values)

    all_values = imputer.transform(drought[features])
    all_values = scaler.transform(all_values)
    return all_values.astype("float32")


def make_sequences(drought, X_rows):
    gaps = drought.groupby("county_fips")["map_date"].diff().dropna().dt.days
    if not gaps.eq(7).all():
        raise ValueError("County weekly dates are not consecutive.")

    groups = drought.groupby("county_fips", sort=False).indices
    total = 0
    for positions in groups.values():
        total += len(positions) - SEQUENCE_LENGTH + 1

    X = np.empty(
        (total, SEQUENCE_LENGTH, X_rows.shape[1]),
        dtype="float32",
    )
    ending_rows = np.empty(total, dtype="int64")

    current = 0
    for positions in groups.values():
        county_values = X_rows[positions]
        windows = np.lib.stride_tricks.sliding_window_view(
            county_values,
            SEQUENCE_LENGTH,
            axis=0,
        ).transpose(0, 2, 1)

        count = len(windows)
        X[current : current + count] = windows
        ending_rows[current : current + count] = positions[SEQUENCE_LENGTH - 1 :]
        current += count

    information = drought.iloc[ending_rows][
        ["state_fips", "year", "map_date", TARGET, WEIGHT]
    ].reset_index(drop=True)

    return X, information


def get_weights(data):
    weights = data[WEIGHT].fillna(0).clip(lower=0)
    if weights.mean() == 0:
        return np.ones(len(data))
    return (weights / weights.mean()).to_numpy()


def train_model(X, information):
    train_rows = (information["year"] <= 2018).to_numpy()
    validation_rows = information["year"].between(2019, 2020).to_numpy()
    backtest_rows = (information["year"] >= 2021).to_numpy()

    X_train = X[train_rows]
    X_validation = X[validation_rows]
    X_backtest = X[backtest_rows]

    train = information[train_rows].reset_index(drop=True)
    validation = information[validation_rows].reset_index(drop=True)
    backtest = information[backtest_rows].reset_index(drop=True)

    target_mean = train[TARGET].mean()
    target_std = train[TARGET].std()
    if target_std == 0:
        target_std = 1

    y_train = ((train[TARGET] - target_mean) / target_std).to_numpy()
    y_validation = ((validation[TARGET] - target_mean) / target_std).to_numpy()

    keras.backend.clear_session()
    keras.utils.set_random_seed(67)

    model = keras.Sequential(
        [
            keras.Input(shape=(SEQUENCE_LENGTH, X.shape[2])),
            layers.SimpleRNN(32),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mean_squared_error",
    )

    stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=get_weights(train),
        validation_data=(X_validation, y_validation, get_weights(validation)),
        epochs=50,
        batch_size=1024,
        callbacks=[stop],
        verbose=1,
    )

    validation_scaled = model.predict(
        X_validation, batch_size=2048, verbose=0
    ).reshape(-1)
    backtest_scaled = model.predict(
        X_backtest, batch_size=2048, verbose=0
    ).reshape(-1)

    validation_predictions = np.clip(
        validation_scaled * target_std + target_mean, 0, 100
    )
    backtest_predictions = np.clip(
        backtest_scaled * target_std + target_mean, 0, 100
    )

    print_results("VALIDATION", validation, validation_predictions)
    print_results("BACKTEST", backtest, backtest_predictions)
    return model


def print_results(name, data, predictions):
    row_mae = mean_absolute_error(data[TARGET], predictions)
    row_rmse = np.sqrt(mean_squared_error(data[TARGET], predictions))
    row_r2 = r2_score(data[TARGET], predictions)

    results = data.copy()
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
    state_rmse = np.sqrt(mean_squared_error(state_year["actual"], state_year["prediction"]))
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
