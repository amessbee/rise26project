"""
download_data.py

Downloads the water and electricity stress CSVs from the public Google
Drive folder into the data/ directory.

Usage:
    pip install gdown
    python download_data.py
"""

import os
import gdown

# Public Google Drive folder containing the dataset CSVs
FOLDER_URL = "https://drive.google.com/drive/folders/1xWyYt1tTwSt4NdBoJceAehcHJfgrC0eX?usp=sharing"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading dataset files into {DATA_DIR} ...\n")

    gdown.download_folder(
        url=FOLDER_URL,
        output=DATA_DIR,
        quiet=False,
        use_cookies=False,
    )

    print("\nDone. Files are in the data/ folder:")
    for f in os.listdir(DATA_DIR):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
