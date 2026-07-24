import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from tensorflow import keras
from tensorflow.keras import layers


FILE_PATH = r"D:\RISE_Project\Feature_CSVs\water_compliance_features_public_water_system_year_2010_2024.csv"
TARGET = "target_health_based_violation_rate_contribution_per_100000_state_residents"
SEQUENCE_LENGTH = 3


def main():
    compliance = pd.read_csv(FILE_PATH)
    compliance = compliance.sort_values(["pwsid", "year"]).reset_index(drop=True)
    compliance, features = add_category_columns(compliance)

    if compliance.duplicated(["pwsid", "year"]).any():
        raise ValueError("Duplicate pwsid-year rows were found.")

    X_rows = preprocess_rows(compliance, features)
    X, information = make_sequences(compliance, X_rows)

    print("Sequence model: RNN")
    print("Consecutive three-year sequences:", len(information))
    print("Features per year:", len(features))

    model = train_model(X, information)
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


def preprocess_rows(compliance, features):
    training_rows = compliance["year"] <= 2018
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()

    imputer.fit(compliance.loc[training_rows, features])
    training_values = imputer.transform(compliance.loc[training_rows, features])
    scaler.fit(training_values)

    all_values = imputer.transform(compliance[features])
    all_values = scaler.transform(all_values)
    return all_values.astype("float32")


def make_sequences(compliance, X_rows):
    groups = compliance.groupby("pwsid", sort=False).indices

    maximum_total = 0
    for positions in groups.values():
        maximum_total += max(0, len(positions) - SEQUENCE_LENGTH + 1)

    X = np.empty(
        (maximum_total, SEQUENCE_LENGTH, X_rows.shape[1]),
        dtype="float32",
    )
    ending_rows = np.empty(maximum_total, dtype="int64")

    current = 0
    for positions in groups.values():
        if len(positions) < SEQUENCE_LENGTH:
            continue

        values = X_rows[positions]
        years = compliance.iloc[positions]["year"].to_numpy()

        windows = np.lib.stride_tricks.sliding_window_view(
            values,
            SEQUENCE_LENGTH,
            axis=0,
        ).transpose(0, 2, 1)

        valid = (
            (years[1:-1] == years[:-2] + 1)
            & (years[2:] == years[1:-1] + 1)
        )

        windows = windows[valid]
        ends = positions[SEQUENCE_LENGTH - 1 :][valid]
        count = len(windows)

        X[current : current + count] = windows
        ending_rows[current : current + count] = ends
        current += count

    X = X[:current]
    ending_rows = ending_rows[:current]

    information = compliance.iloc[ending_rows][
        ["state_fips", "year", TARGET]
    ].reset_index(drop=True)

    return X, information


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

    train_log = np.log1p(train[TARGET])
    validation_log = np.log1p(validation[TARGET])
    target_mean = train_log.mean()
    target_std = train_log.std()
    if target_std == 0:
        target_std = 1

    y_train = ((train_log - target_mean) / target_std).to_numpy()
    y_validation = ((validation_log - target_mean) / target_std).to_numpy()

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
        validation_data=(X_validation, y_validation),
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

    validation_log_predictions = np.clip(
        validation_scaled * target_std + target_mean, 0, train_log.max()
    )
    backtest_log_predictions = np.clip(
        backtest_scaled * target_std + target_mean, 0, train_log.max()
    )

    validation_predictions = np.expm1(validation_log_predictions)
    backtest_predictions = np.expm1(backtest_log_predictions)

    print("Validation system-year rows covered:", len(validation))
    print_results("VALIDATION", validation, validation_predictions)
    print("Backtest system-year rows covered:", len(backtest))
    print_results("BACKTEST", backtest, backtest_predictions)
    return model


def print_results(name, data, predictions):
    system_mae = mean_absolute_error(data[TARGET], predictions)
    system_rmse = np.sqrt(mean_squared_error(data[TARGET], predictions))
    system_r2 = r2_score(data[TARGET], predictions)

    results = data.copy()
    results["prediction"] = predictions
    state_year = results.groupby(["state_fips", "year"])[
        [TARGET, "prediction"]
    ].sum()

    state_mae = mean_absolute_error(state_year[TARGET], state_year["prediction"])
    state_rmse = np.sqrt(mean_squared_error(state_year[TARGET], state_year["prediction"]))
    state_r2 = r2_score(state_year[TARGET], state_year["prediction"])

    print("\n" + name + " RESULTS")
    print("System-year MAE:", round(system_mae, 6))
    print("System-year RMSE:", round(system_rmse, 6))
    print("System-year R2:", round(system_r2, 3))
    print("Covered-system state-year MAE:", round(state_mae, 3))
    print("Covered-system state-year RMSE:", round(state_rmse, 3))
    print("Covered-system state-year R2:", round(state_r2, 3))


if __name__ == "__main__":
    main()
