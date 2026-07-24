import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


electricity = pd.read_csv(
    r"D:\RISE_Project\Feature_CSVs"
    r"\electricity_features_final_sequence_2013_2024.csv"
)


# These columns will not be used as model features.
not_features = [
    "sequence_id",
    "utility_number",
    "utility_name",
    "state_fips",
    "state_name",
    "input_year",
    "target_year",
    "target_saidi_minutes_per_customer",
    "target_saifi_interruptions_per_customer",
    "sample_weight_reporting_customers",
]


# Select the numerical feature columns.
number_columns = electricity.select_dtypes(
    include="number"
).columns

features = []

for column in number_columns:
    if column not in not_features:
        features.append(column)


def main():
    print(
        "Rows in electricity file:",
        len(electricity),
    )

    print(
        "Features being used:",
        len(features),
    )

    print("\nSAIDI MODEL")

    saidi_model = train_pipeline(
        "target_saidi_minutes_per_customer"
    )

    print("\nSAIFI MODEL")

    saifi_model = train_pipeline(
        "target_saifi_interruptions_per_customer"
    )

    return saidi_model, saifi_model


def split_data(target):
    # Remove rows where the required target is missing.
    data = electricity.dropna(
        subset=[target]
    ).copy()

    # Use the earlier years for training.
    train = data[
        data["target_year"] <= 2018
    ]

    # Use 2019-2020 to choose ANN settings.
    validation = data[
        data["target_year"].between(
            2019,
            2020,
        )
    ]

    # Keep 2021-2024 untouched for final testing.
    test = data[
        data["target_year"] >= 2021
    ]

    return train, validation, test


def make_ann(layers, solver):
    model = make_pipeline(
        # Replace missing features using training medians.
        SimpleImputer(
            strategy="median"
        ),

        # Put all input features on similar scales.
        StandardScaler(),

        # Create the artificial neural network.
        MLPRegressor(
            hidden_layer_sizes=layers,
            activation="relu",
            loss="squared_error",
            solver=solver,
            max_iter=500,
            early_stopping=True,
            random_state=67,
        ),
    )

    return model


def predict(
    model,
    training_data,
    prediction_data,
    target,
):
    # SAIDI contains very large extreme values.
    # Train using log1p(SAIDI).
    if target == "target_saidi_minutes_per_customer":

        training_target = np.log1p(
            training_data[target]
        )

        model.fit(
            training_data[features],
            training_target,
        )

        log_predictions = model.predict(
            prediction_data[features]
        )

        # Do not allow the ANN to extrapolate beyond
        # the log range observed during training.
        log_predictions = np.clip(
            log_predictions,
            0,
            training_target.max(),
        )

        # Convert predictions back into SAIDI minutes.
        predictions = np.expm1(
            log_predictions
        )

    # SAIFI is much less extreme, so mean scaling is used.
    else:
        target_scale = training_data[
            target
        ].mean()

        scaled_target = (
            training_data[target]
            / target_scale
        )

        model.fit(
            training_data[features],
            scaled_target,
        )

        predictions = (
            model.predict(
                prediction_data[features]
            )
            * target_scale
        )

        # Keep SAIFI predictions inside the target
        # range observed during training.
        predictions = np.clip(
            predictions,
            0,
            training_data[target].max(),
        )

    # SAIDI and SAIFI cannot be negative.
    predictions = np.clip(
        predictions,
        0,
        None,
    )

    return predictions


def evaluate(
    data,
    predictions,
    target,
):
    results = data[
        ["sequence_id", target]
    ].copy()

    results["prediction"] = predictions

    # Each sequence has 12 monthly input rows.
    # Average their predictions to produce one
    # annual prediction per utility-state-year.
    annual_results = (
        results.groupby("sequence_id").mean()
    )

    actual = annual_results[target]
    predicted = annual_results["prediction"]

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


def train_pipeline(target):
    train, validation, test = split_data(
        target
    )

    print(
        "Training rows:",
        len(train),
    )

    print(
        "Validation rows:",
        len(validation),
    )

    print(
        "Testing rows:",
        len(test),
    )

    # One number means one hidden layer.
    # Two numbers mean two hidden layers.
    layer_options = [
        (32,),
        (64,),
        (64, 32),
    ]

    solver_options = [
        "adam",
        "sgd",
    ]

    best_rmse = float("inf")
    best_layers = None
    best_solver = None

    # Test every layer and optimizer combination.
    for layers in layer_options:
        for solver in solver_options:

            model = make_ann(
                layers,
                solver,
            )

            validation_predictions = predict(
                model,
                train,
                validation,
                target,
            )

            mae, rmse, r2 = evaluate(
                validation,
                validation_predictions,
                target,
            )

            print()
            print("Layers:", layers)
            print("Loss: squared_error")
            print("Solver:", solver)

            print(
                "Validation MAE:",
                round(mae, 2),
            )

            print(
                "Validation RMSE:",
                round(rmse, 2),
            )

            print(
                "Validation R2:",
                round(r2, 3),
            )

            # The lowest validation RMSE wins.
            if rmse < best_rmse:
                best_rmse = rmse
                best_layers = layers
                best_solver = solver

    print()
    print("BEST SETTINGS")
    print("Layers:", best_layers)
    print("Loss: squared_error")
    print("Solver:", best_solver)
    print(
        "Best validation RMSE:",
        round(best_rmse, 2),
    )

    # After choosing the settings, combine
    # training and validation data.
    final_training_data = pd.concat(
        [train, validation]
    )

    final_model = make_ann(
        best_layers,
        best_solver,
    )

    test_predictions = predict(
        final_model,
        final_training_data,
        test,
        target,
    )

    test_mae, test_rmse, test_r2 = evaluate(
        test,
        test_predictions,
        target,
    )

    print()
    print("FINAL TEST RESULTS")

    print(
        "MAE:",
        round(test_mae, 2),
    )

    print(
        "RMSE:",
        round(test_rmse, 2),
    )

    print(
        "R2:",
        round(test_r2, 3),
    )

    return final_model


if __name__ == "__main__":
    main()