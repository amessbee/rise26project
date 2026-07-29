import json
import math
import os

import pandas as pd


PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.join(PROJECT_FOLDER, "saved_models")
HISTORY_FILE = os.path.join(MODEL_FOLDER, "state_model_history.csv")
INFORMATION_FILE = os.path.join(MODEL_FOLDER, "project_metadata.json")


REQUIRED_HISTORY_COLUMNS = [
    "state_fips",
    "state_name",
    "state_abbreviation",
    "year",
    "actual_saidi",
    "predicted_saidi",
    "actual_saifi",
    "predicted_saifi",
    "actual_drought",
    "predicted_drought",
    "actual_compliance",
    "predicted_compliance",
]

REQUIRED_INFORMATION_KEYS = [
    "latest_observed_year",
    "normalization_boundaries",
    "projection",
    "backtest_rmse",
    "best_models",
    "main_historical_model_inputs",
    "important_warning",
]


def main():
    try:
        history = pd.read_csv(
            HISTORY_FILE,
            dtype={"state_fips": str},
        )

        with open(INFORMATION_FILE, "r", encoding="utf-8") as file:
            information = json.load(file)

    except FileNotFoundError:
        print("The trained model files were not found.")
        print("Run this first: py -3.13 train_models_once.py")
        return

    try:
        validate_files(history, information)
    except ValueError as error:
        print("The trained model files are incomplete or invalid.")
        print(error)
        print("Rebuild them with: py -3.13 train_models_once.py --force")
        return

    history["state_fips"] = history["state_fips"].astype(str).str.zfill(2)
    history["year"] = pd.to_numeric(history["year"], errors="coerce")

    if history["year"].isna().any():
        print("The history file contains invalid year values.")
        print("Rebuild it with: py -3.13 train_models_once.py --force")
        return

    history["year"] = history["year"].astype(int)

    latest_year = int(information["latest_observed_year"])

    print()
    print("U.S. ELECTRICITY AND WATER STRESS PROJECTOR")
    print(f"Latest observed year: {latest_year}")

    state_fips, state_name = select_state(history)
    future_year = select_year(latest_year)

    state_history = history.loc[history["state_fips"] == state_fips].sort_values(
        "year"
    )

    predicted_saidi = predict_normal_target(
        state_history,
        "actual_saidi",
        "predicted_saidi",
        latest_year,
        future_year,
    )

    predicted_saifi = predict_normal_target(
        state_history,
        "actual_saifi",
        "predicted_saifi",
        latest_year,
        future_year,
    )

    predicted_drought = predict_drought(
        state_history,
        latest_year,
        future_year,
    )

    predicted_compliance = predict_normal_target(
        state_history,
        "actual_compliance",
        "predicted_compliance",
        latest_year,
        future_year,
    )

    boundaries = information["normalization_boundaries"]

    duration_stress = normalize(
        predicted_saidi,
        boundaries["saidi"]["low"],
        boundaries["saidi"]["high"],
    )

    frequency_stress = normalize(
        predicted_saifi,
        boundaries["saifi"]["low"],
        boundaries["saifi"]["high"],
    )

    drought_stress = predicted_drought

    compliance_stress = normalize(
        predicted_compliance,
        boundaries["compliance"]["low"],
        boundaries["compliance"]["high"],
    )

    electricity_stress = (duration_stress + frequency_stress) / 2
    water_stress = (drought_stress + compliance_stress) / 2

    latest_saidi = get_latest(state_history, "actual_saidi")
    latest_saifi = get_latest(state_history, "actual_saifi")
    latest_drought = get_latest(state_history, "actual_drought")
    latest_compliance = get_latest(state_history, "actual_compliance")

    latest_electricity_stress = (
        normalize(
            latest_saidi,
            boundaries["saidi"]["low"],
            boundaries["saidi"]["high"],
        )
        + normalize(
            latest_saifi,
            boundaries["saifi"]["low"],
            boundaries["saifi"]["high"],
        )
    ) / 2

    latest_water_stress = (
        latest_drought
        + normalize(
            latest_compliance,
            boundaries["compliance"]["low"],
            boundaries["compliance"]["high"],
        )
    ) / 2

    electricity_change = electricity_stress - latest_electricity_stress
    water_change = water_stress - latest_water_stress

    number_of_years = future_year - latest_year
    errors = information["backtest_rmse"]

    electricity_low, electricity_high = electricity_range(
        predicted_saidi,
        predicted_saifi,
        number_of_years,
        errors,
        boundaries,
    )

    water_low, water_high = water_range(
        predicted_drought,
        predicted_compliance,
        number_of_years,
        errors,
        boundaries,
    )

    print()
    print("-------------------------------------------")
    print(f"{state_name.upper()} STRESS PROJECTION FOR {future_year}")
    print("-------------------------------------------")

    print()
    print("ELECTRICITY")
    print(f"Predicted SAIDI: {predicted_saidi:.2f} minutes per customer")
    print(f"Predicted SAIFI: {predicted_saifi:.3f} interruptions per customer")
    print(f"Duration stress: {duration_stress:.1f} / 100")
    print(f"Frequency stress: {frequency_stress:.1f} / 100")
    print(f"FINAL ELECTRICITY STRESS: {electricity_stress:.1f} / 100")
    print(f"Change from {latest_year}: {electricity_change:+.1f} points")
    print(f"Approximate range: {electricity_low:.1f} to {electricity_high:.1f}")

    print()
    print("WATER")
    print(f"Predicted drought severity: {predicted_drought:.2f} / 100")
    print(
        "Predicted compliance violation rate: "
        f"{predicted_compliance:.3f} per 100,000 residents"
    )
    print(f"Drought stress: {drought_stress:.1f} / 100")
    print(f"Compliance stress: {compliance_stress:.1f} / 100")
    print(f"FINAL WATER STRESS: {water_stress:.1f} / 100")
    print(f"Change from {latest_year}: {water_change:+.1f} points")
    print(f"Approximate range: {water_low:.1f} to {water_high:.1f}")

    inputs = information["main_historical_model_inputs"]
    electricity_input = first_input_name(inputs, "saidi")
    water_input = first_input_name(inputs, "drought")

    print()
    print("MAIN HISTORICAL MODEL INPUTS")
    print(f"Electricity: {electricity_input}")
    print(f"Water: {water_input}")

    print()
    print(f"The program calculated {number_of_years} one-year predictions.")
    print("Each predicted year was used to calculate the next year.")
    print("This is a baseline projection because future weather,")
    print("population and infrastructure are not known yet.")


def validate_files(history, information):
    missing_columns = [
        column for column in REQUIRED_HISTORY_COLUMNS if column not in history.columns
    ]
    if missing_columns:
        raise ValueError(
            "state_model_history.csv is missing: " + ", ".join(missing_columns)
        )

    missing_keys = [
        key for key in REQUIRED_INFORMATION_KEYS if key not in information
    ]
    if missing_keys:
        raise ValueError(
            "project_metadata.json is missing: " + ", ".join(missing_keys)
        )

    if history.empty:
        raise ValueError("state_model_history.csv contains no rows.")

    for target in ["saidi", "saifi", "drought", "compliance"]:
        actual_column = "actual_" + target
        prediction_column = "predicted_" + target

        if history[actual_column].dropna().empty:
            raise ValueError(actual_column + " contains no observed values.")

        if history[prediction_column].dropna().empty:
            raise ValueError(prediction_column + " contains no model signals.")

        prediction_values = pd.to_numeric(
            history[prediction_column].dropna(),
            errors="coerce",
        )

        if prediction_values.isna().any():
            raise ValueError(prediction_column + " contains invalid values.")

    boundaries = information["normalization_boundaries"]
    for target in ["saidi", "saifi", "compliance"]:
        if target not in boundaries:
            raise ValueError("Missing normalization boundaries for " + target + ".")

        low = float(boundaries[target]["low"])
        high = float(boundaries[target]["high"])

        if not math.isfinite(low) or not math.isfinite(high) or high <= low:
            raise ValueError("Invalid normalization boundaries for " + target + ".")

    errors = information["backtest_rmse"]
    for target in ["saidi", "saifi", "drought", "compliance"]:
        value = float(errors[target])
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid backtest RMSE for " + target + ".")


def first_input_name(inputs, target):
    values = inputs.get(target, [])
    if len(values) == 0:
        return "not available"
    return str(values[0]).replace("_", " ")


def select_state(history):
    state_table = history.loc[
        history["state_name"].notna(),
        [
            "state_fips",
            "state_name",
            "state_abbreviation",
        ],
    ].copy()

    state_table["state_name"] = state_table["state_name"].fillna("").astype(str)
    state_table["state_abbreviation"] = (
        state_table["state_abbreviation"].fillna("").astype(str)
    )
    state_table = state_table.drop_duplicates(subset=["state_fips"])

    if state_table.empty:
        raise ValueError("No state names were available in the history file.")

    state_found = False

    while state_found == False:
        state = input("\nEnter a U.S. state name or abbreviation: ")
        state = state.lower().strip().replace(".", "")

        if state in ["washington dc", "washington d c", "dc"]:
            state = "district of columbia"

        matches = state_table.loc[
            (state_table["state_name"].str.lower() == state)
            | (state_table["state_abbreviation"].str.lower() == state)
        ]

        if matches.empty:
            print("State not found. Examples: Texas, texas, TX or tx.")
        else:
            state_found = True

    state_fips = matches.iloc[0]["state_fips"]
    state_name = matches.iloc[0]["state_name"]

    return state_fips, state_name


def select_year(latest_year):
    correct_year = False

    while correct_year == False:
        year = input("Enter a future year using four digits, such as 2067: ").strip()

        if not year.isnumeric():
            print("The year must only contain numbers.")

        elif len(year) != 4:
            print("The year must contain exactly four digits.")

        elif int(year) <= latest_year:
            print(f"Enter a future year after {latest_year}.")

        elif int(year) > 2100:
            print("Enter a year no later than 2100.")

        else:
            correct_year = True

    return int(year)


def target_history_rows(state_history, actual_column, prediction_column):
    target_history = state_history.loc[
        state_history[actual_column].notna(),
        [
            "year",
            actual_column,
            prediction_column,
        ],
    ].copy()

    if target_history.empty:
        raise ValueError("No history is available for " + actual_column + ".")

    target_history[actual_column] = pd.to_numeric(
        target_history[actual_column],
        errors="coerce",
    )

    target_history[prediction_column] = pd.to_numeric(
        target_history[prediction_column],
        errors="coerce",
    )

    target_history.dropna(subset=[actual_column], inplace=True)

    if target_history.empty:
        raise ValueError("No numeric history is available for " + actual_column + ".")

    target_history["model_signal"] = target_history[prediction_column].fillna(
        target_history[actual_column]
    )

    return target_history


def predict_normal_target(
    state_history,
    actual_column,
    prediction_column,
    latest_year,
    future_year,
):
    target_history = target_history_rows(
        state_history,
        actual_column,
        prediction_column,
    )

    prediction = float(target_history.iloc[-1][actual_column])
    historical_typical_value = float(target_history[actual_column].median())
    recent_history = target_history.tail(5)
    recent_changes = recent_history["model_signal"].diff().dropna()

    if len(recent_changes) > 0:
        yearly_change = float(recent_changes.median())
    else:
        yearly_change = 0

    for year in range(latest_year + 1, future_year + 1):
        prediction = prediction + yearly_change
        prediction = (prediction * 0.8) + (historical_typical_value * 0.2)

        if prediction < 0:
            prediction = 0

        yearly_change = yearly_change * 0.5

    if not math.isfinite(prediction):
        raise ValueError("The projection became nonfinite for " + actual_column + ".")

    return prediction


def predict_drought(
    state_history,
    latest_year,
    future_year,
):
    drought_history = target_history_rows(
        state_history,
        "actual_drought",
        "predicted_drought",
    )

    prediction = float(drought_history.iloc[-1]["actual_drought"])
    historical_average = float(drought_history["actual_drought"].mean())
    recent_history = drought_history.tail(5)
    recent_changes = recent_history["model_signal"].diff().dropna()

    if len(recent_changes) > 0:
        yearly_change = float(recent_changes.median())
    else:
        yearly_change = 0

    for year in range(latest_year + 1, future_year + 1):
        prediction = prediction + yearly_change
        prediction = (prediction * 0.8) + (historical_average * 0.2)
        prediction = min(100, max(0, prediction))
        yearly_change = yearly_change * 0.5

    if not math.isfinite(prediction):
        raise ValueError("The drought projection became nonfinite.")

    return prediction


def get_latest(state_history, column):
    values = pd.to_numeric(
        state_history.loc[state_history[column].notna(), column],
        errors="coerce",
    ).dropna()

    if values.empty:
        raise ValueError("No observed value is available for " + column + ".")

    return float(values.iloc[-1])


def normalize(value, low, high):
    low = float(low)
    high = float(high)

    if high <= low:
        raise ValueError("A normalization boundary has high <= low.")

    score = 100 * (value - low) / (high - low)
    return min(100, max(0, score))


def electricity_range(
    saidi,
    saifi,
    number_of_years,
    errors,
    boundaries,
):
    widening = number_of_years**0.5
    saidi_error = 1.96 * float(errors["saidi"]) * widening
    saifi_error = 1.96 * float(errors["saifi"]) * widening

    low_duration = normalize(
        max(0, saidi - saidi_error),
        boundaries["saidi"]["low"],
        boundaries["saidi"]["high"],
    )

    high_duration = normalize(
        saidi + saidi_error,
        boundaries["saidi"]["low"],
        boundaries["saidi"]["high"],
    )

    low_frequency = normalize(
        max(0, saifi - saifi_error),
        boundaries["saifi"]["low"],
        boundaries["saifi"]["high"],
    )

    high_frequency = normalize(
        saifi + saifi_error,
        boundaries["saifi"]["low"],
        boundaries["saifi"]["high"],
    )

    low = (low_duration + low_frequency) / 2
    high = (high_duration + high_frequency) / 2

    return min(low, high), max(low, high)


def water_range(
    drought,
    compliance,
    number_of_years,
    errors,
    boundaries,
):
    widening = number_of_years**0.5
    drought_error = 1.96 * float(errors["drought"]) * widening
    compliance_error = 1.96 * float(errors["compliance"]) * widening

    low_drought = max(0, drought - drought_error)
    high_drought = min(100, drought + drought_error)

    low_compliance = normalize(
        max(0, compliance - compliance_error),
        boundaries["compliance"]["low"],
        boundaries["compliance"]["high"],
    )

    high_compliance = normalize(
        compliance + compliance_error,
        boundaries["compliance"]["low"],
        boundaries["compliance"]["high"],
    )

    low = (low_drought + low_compliance) / 2
    high = (high_drought + high_compliance) / 2

    return min(low, high), max(low, high)


if __name__ == "__main__":
    main()
