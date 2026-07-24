import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from tensorflow import keras
from tensorflow.keras import layers

# Change this to "LSTM" in the LSTM file.
MODEL_TYPE = "RNN"


FILE_PATH = (
    r"D:\RISE_Project\Feature_CSVs"
    r"\electricity_features_final_sequence_2013_2024.csv"
)

SAIDI_TARGET = "target_saidi_minutes_per_customer"

SAIFI_TARGET = "target_saifi_interruptions_per_customer"

WEIGHT_COLUMN = "sample_weight_reporting_customers"


def main():
    electricity = pd.read_csv(FILE_PATH)

    sequence_information, X, features = prepare_sequences(electricity)

    print("Model type:", MODEL_TYPE)
    print(
        "Annual sequences:",
        len(sequence_information),
    )
    print(
        "Features per month:",
        len(features),
    )
    print(
        "Months per sequence:",
        X.shape[1],
    )

    print("\nSAIDI MODEL")

    saidi_model = train_pipeline(
        sequence_information,
        X,
        SAIDI_TARGET,
    )

    print("\nSAIFI MODEL")

    saifi_model = train_pipeline(
        sequence_information,
        X,
        SAIFI_TARGET,
    )

    return saidi_model, saifi_model


def prepare_sequences(electricity):
    # Add state, ownership and reporting standard
    # as categorical dummy columns.
    category_data = electricity[
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

    electricity = pd.concat(
        [
            electricity,
            category_columns,
        ],
        axis=1,
    )

    # Columns that must not be given to the model.
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

    features = []

    for column in number_columns:
        if column not in not_features:
            features.append(column)

    # Ensure the 12 months are in order.
    electricity = electricity.sort_values(
        [
            "sequence_id",
            "month",
        ]
    ).reset_index(drop=True)

    sequence_sizes = electricity.groupby("sequence_id").size()

    if sequence_sizes.min() != 12:
        raise ValueError("At least one sequence has fewer than 12 months.")

    if sequence_sizes.max() != 12:
        raise ValueError("At least one sequence has more than 12 months.")

    information_columns = [
        "utility_number",
        "state_fips",
        "state_name",
        "target_year",
        SAIDI_TARGET,
        SAIFI_TARGET,
        WEIGHT_COLUMN,
    ]

    sequence_information = (
        electricity.groupby(
            "sequence_id",
            sort=False,
        )[information_columns]
        .first()
        .reset_index()
    )

    # Convert the monthly rows into:
    # sequences × 12 months × features.
    X = electricity[features].to_numpy(dtype="float32")

    X = X.reshape(
        len(sequence_information),
        12,
        len(features),
    )

    return sequence_information, X, features


def get_weights(information):
    weights = pd.to_numeric(
        information[WEIGHT_COLUMN],
        errors="coerce",
    )

    median_weight = weights.median()

    if pd.isna(median_weight):
        median_weight = 1

    weights = weights.fillna(median_weight)

    weights = weights.clip(lower=1)

    weights = weights / weights.mean()

    return weights.to_numpy()


def prepare_input_data(
    X_train,
    X_validation,
    X_backtest,
):
    number_of_features = X_train.shape[2]

    # Convert the 3D sequence temporarily into
    # 2D so scikit-learn can preprocess it.
    train_flat = X_train.reshape(
        -1,
        number_of_features,
    )

    validation_flat = X_validation.reshape(
        -1,
        number_of_features,
    )

    backtest_flat = X_backtest.reshape(
        -1,
        number_of_features,
    )

    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )

    scaler = StandardScaler()

    train_flat = imputer.fit_transform(train_flat)

    validation_flat = imputer.transform(validation_flat)

    backtest_flat = imputer.transform(backtest_flat)

    train_flat = scaler.fit_transform(train_flat)

    validation_flat = scaler.transform(validation_flat)

    backtest_flat = scaler.transform(backtest_flat)

    final_number_of_features = train_flat.shape[1]

    X_train = train_flat.reshape(
        len(X_train),
        12,
        final_number_of_features,
    ).astype("float32")

    X_validation = validation_flat.reshape(
        len(X_validation),
        12,
        final_number_of_features,
    ).astype("float32")

    X_backtest = backtest_flat.reshape(
        len(X_backtest),
        12,
        final_number_of_features,
    ).astype("float32")

    return (
        X_train,
        X_validation,
        X_backtest,
    )


def prepare_target(
    train_information,
    validation_information,
    target,
):
    train_target = train_information[target].to_numpy(dtype=float)

    validation_target = validation_information[target].to_numpy(dtype=float)

    # Log-transform SAIDI.
    if target == SAIDI_TARGET:
        train_target = np.log1p(train_target)

        validation_target = np.log1p(validation_target)

    target_mean = train_target.mean()
    target_standard_deviation = train_target.std()

    if target_standard_deviation == 0:
        target_standard_deviation = 1

    train_target = (train_target - target_mean) / target_standard_deviation

    validation_target = (validation_target - target_mean) / target_standard_deviation

    return (
        train_target,
        validation_target,
        target_mean,
        target_standard_deviation,
    )


def make_model(number_of_features):
    keras.backend.clear_session()
    keras.utils.set_random_seed(67)

    model = keras.Sequential()

    model.add(
        keras.Input(
            shape=(
                12,
                number_of_features,
            )
        )
    )

    if MODEL_TYPE == "RNN":
        model.add(
            layers.SimpleRNN(
                64,
                return_sequences=True,
            )
        )

        model.add(layers.SimpleRNN(32))

    elif MODEL_TYPE == "LSTM":
        model.add(
            layers.LSTM(
                64,
                return_sequences=True,
            )
        )

        model.add(layers.LSTM(32))

    else:
        raise ValueError("MODEL_TYPE must be RNN or LSTM.")

    model.add(
        layers.Dense(
            32,
            activation="relu",
        )
    )

    model.add(layers.Dropout(0.1))

    model.add(layers.Dense(1))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss="mean_squared_error",
    )

    return model


def convert_predictions(
    scaled_predictions,
    training_information,
    target,
    target_mean,
    target_standard_deviation,
):
    transformed_predictions = (
        scaled_predictions * target_standard_deviation + target_mean
    )

    if target == SAIDI_TARGET:
        largest_training_value = np.log1p(training_information[target]).max()

        transformed_predictions = np.clip(
            transformed_predictions,
            0,
            largest_training_value,
        )

        predictions = np.expm1(transformed_predictions)

    else:
        largest_training_value = training_information[target].max()

        predictions = np.clip(
            transformed_predictions,
            0,
            largest_training_value,
        )

    return predictions


def calculate_metrics(
    information,
    predictions,
    target,
):
    actual = information[target].to_numpy(dtype=float)

    utility_mae = mean_absolute_error(
        actual,
        predictions,
    )

    utility_rmse = np.sqrt(
        mean_squared_error(
            actual,
            predictions,
        )
    )

    utility_r2 = r2_score(
        actual,
        predictions,
    )

    # Customer-weighted state results.
    results = information[
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

    state_results["actual"] = state_results["actual_weighted"] / state_results["weight"]

    state_results["prediction"] = (
        state_results["prediction_weighted"] / state_results["weight"]
    )

    state_mae = mean_absolute_error(
        state_results["actual"],
        state_results["prediction"],
    )

    state_rmse = np.sqrt(
        mean_squared_error(
            state_results["actual"],
            state_results["prediction"],
        )
    )

    state_r2 = r2_score(
        state_results["actual"],
        state_results["prediction"],
    )

    return (
        utility_mae,
        utility_rmse,
        utility_r2,
        state_mae,
        state_rmse,
        state_r2,
    )


def display_results(
    information,
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
        information,
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


def train_pipeline(
    sequence_information,
    X,
    target,
):
    # Remove sequences with missing targets.
    available_target = sequence_information[target].notna().to_numpy()

    information = sequence_information[available_target].copy().reset_index(drop=True)

    X = X[available_target]

    train_rows = (information["target_year"] <= 2018).to_numpy()

    validation_rows = (information["target_year"].between(2019, 2020)).to_numpy()

    backtest_rows = (information["target_year"] >= 2021).to_numpy()

    train_information = information[train_rows].copy().reset_index(drop=True)

    validation_information = information[validation_rows].copy().reset_index(drop=True)

    backtest_information = information[backtest_rows].copy().reset_index(drop=True)

    X_train = X[train_rows]
    X_validation = X[validation_rows]
    X_backtest = X[backtest_rows]

    (
        X_train,
        X_validation,
        X_backtest,
    ) = prepare_input_data(
        X_train,
        X_validation,
        X_backtest,
    )

    (
        y_train,
        y_validation,
        target_mean,
        target_standard_deviation,
    ) = prepare_target(
        train_information,
        validation_information,
        target,
    )

    training_weights = get_weights(train_information)

    validation_weights = get_weights(validation_information)

    model = make_model(X_train.shape[2])

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=training_weights,
        validation_data=(
            X_validation,
            y_validation,
            validation_weights,
        ),
        epochs=150,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=1,
    )

    scaled_predictions = model.predict(
        X_backtest,
        verbose=0,
    ).reshape(-1)

    predictions = convert_predictions(
        scaled_predictions,
        train_information,
        target,
        target_mean,
        target_standard_deviation,
    )

    display_results(
        backtest_information,
        predictions,
        target,
    )

    return model


if __name__ == "__main__":
    main()
