"""
Download the three frozen model-ready CSVs used by train_models_once.py.

The files are downloaded from the project's public Google Drive folder into
Reproducibility/data/. The downloader verifies that every required filename is
present and moves files out of any nested Drive folder into data/.

Usage:
    python -m pip install -r requirements.txt
    python csv_downloader.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import gdown


FOLDER_URL = (
    "https://drive.google.com/drive/folders/"
    "1xWyYt1tTwSt4NdBoJceAehcHJfgrC0eX?usp=sharing"
)

PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = PROJECT_FOLDER / "data"

REQUIRED_FILES = (
    "electricity_features_final_sequence_2013_2024.csv",
    "water_drought_features_county_week_2010_2024.csv",
    "water_compliance_features_public_water_system_year_2010_2024.csv",
)


def locate_downloaded_file(filename: str) -> Path:
    matches = [
        path
        for path in DATA_FOLDER.rglob(filename)
        if path.is_file()
    ]
    if not matches:
        raise FileNotFoundError(
            f"The Google Drive download did not contain {filename}."
        )
    if len(matches) > 1:
        locations = "\n".join(str(path) for path in matches)
        raise RuntimeError(
            f"Multiple copies of {filename} were downloaded:\n{locations}"
        )
    return matches[0]


def main() -> None:
    DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    print("Downloading the three model-ready datasets...")
    print(f"Destination: {DATA_FOLDER}")
    gdown.download_folder(
        url=FOLDER_URL,
        output=str(DATA_FOLDER),
        quiet=False,
        use_cookies=False,
    )

    print("\nVerifying required files...")
    for filename in REQUIRED_FILES:
        downloaded = locate_downloaded_file(filename)
        destination = DATA_FOLDER / filename
        if downloaded.resolve() != destination.resolve():
            if destination.exists():
                destination.unlink()
            shutil.move(str(downloaded), str(destination))

        size_gib = destination.stat().st_size / (1024**3)
        print(f"  OK  {filename}  ({size_gib:.2f} GiB)")

    print("\nAll model-ready datasets are available.")
    print("Retrain with: python train_models_once.py --force")


if __name__ == "__main__":
    main()
