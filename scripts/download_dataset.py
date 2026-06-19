from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import kagglehub
import pandas as pd


DATASET = "sgpjesus/bank-account-fraud-dataset-neurips-2022"
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the real BAF dataset using KaggleHub.")
    parser.add_argument("--sample-size", type=int, default=50_000, help="Number of rows to sample for local development.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--file", type=str, default="Base.csv", help="Preferred CSV file inside the dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    downloaded_path = Path(kagglehub.dataset_download(DATASET))
    print(f"Downloaded Kaggle dataset to: {downloaded_path}")

    csv_files = sorted(downloaded_path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the downloaded dataset.")

    print("CSV files found:")
    for file in csv_files:
        print(f"- {file.name}")

    preferred = next((f for f in csv_files if f.name.lower() == args.file.lower()), None)
    selected_file = preferred or csv_files[0]
    print(f"Using real dataset file: {selected_file}")

    raw_target = RAW_DIR / selected_file.name
    shutil.copy(selected_file, raw_target)

    df = pd.read_csv(raw_target)
    if "fraud_bool" not in df.columns:
        raise ValueError("Expected label column 'fraud_bool' was not found. Check the selected BAF file.")

    sample_size = min(args.sample_size, len(df))
    sample_df = df.sample(sample_size, random_state=args.seed) if sample_size < len(df) else df
    sample_df = sample_df.reset_index(drop=True)
    sample_df.insert(0, "application_id", [str(i) for i in range(len(sample_df))])

    output_path = PROCESSED_DIR / "baf_base_sample.csv"
    sample_df.to_csv(output_path, index=False)

    print(f"Saved processed real dataset sample to: {output_path}")
    print(f"Rows: {len(sample_df):,}")
    print(f"Columns: {len(sample_df.columns):,}")
    print(f"Fraud rate: {sample_df['fraud_bool'].mean():.6f}")


if __name__ == "__main__":
    main()
