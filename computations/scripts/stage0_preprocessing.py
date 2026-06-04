# stage0_preprocessing.py
#
# Stage 0: Data loading, cleaning, feature engineering, and normalisation.
# Produces a single clean .pkl file consumed by stage1_clustering.py.
#
# Run first: python stage0_preprocessing.py

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from rich.console import Console

CONSOLE = Console()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RAW_TRAIN_PATH   = "ashrae-energy-prediction/train.csv"
RAW_META_PATH    = "ashrae-energy-prediction/building_metadata.csv"
OUTPUT_DIR       = "outputs"
PROCESSED_DATA_PATH = os.path.join(OUTPUT_DIR, "preprocessed_daily_profiles.pkl")


def run_preprocessing():
    """Loads, cleans, filters, and transforms the raw ASHRAE data into daily profiles."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(PROCESSED_DATA_PATH):
        CONSOLE.print(
            f"[bold yellow]Cache found:[/] Preprocessed data already exists at "
            f"[cyan]{PROCESSED_DATA_PATH}[/cyan]. Skipping."
        )
        return

    # Use tqdm to track each named preprocessing step
    steps = [
        "Loading raw CSV files",
        "Filtering electricity meters",
        "Merging building metadata",
        "Parsing timestamps & engineering features",
        "Clipping negatives & dropping NaNs",
        "Pivoting to daily 24-hour profiles",
        "Dropping incomplete days",
        "Applying Min-Max normalisation",
        "Saving preprocessed profiles",
    ]

    with tqdm(total=len(steps), desc="Stage 0 — Preprocessing", unit="step",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:

        # Step 1 — Load
        pbar.set_description(f"Stage 0 — {steps[0]}")
        try:
            df_train = pd.read_csv(RAW_TRAIN_PATH)
            df_meta  = pd.read_csv(RAW_META_PATH)
        except FileNotFoundError as e:
            CONSOLE.print(
                "[bold red]Error:[/] Raw data not found. Place [cyan]train.csv[/cyan] and "
                "[cyan]building_metadata.csv[/cyan] in [cyan]ashrae-energy-prediction/[/cyan]."
            )
            CONSOLE.print(f"Details: {e}")
            return
        pbar.update(1)

        # Step 2 — Filter electricity meter (meter == 0)
        pbar.set_description(f"Stage 0 — {steps[1]}")
        df_train = df_train[df_train["meter"] == 0].copy()
        pbar.update(1)

        # Step 3 — Merge metadata
        pbar.set_description(f"Stage 0 — {steps[2]}")
        df = pd.merge(df_train, df_meta, on="building_id", how="left")
        pbar.update(1)

        # Step 4 — Timestamp parsing & feature engineering
        pbar.set_description(f"Stage 0 — {steps[3]}")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"]      = df["timestamp"].dt.hour
        df["date"]      = df["timestamp"].dt.date
        pbar.update(1)

        # Step 5 — Clip negatives & drop NaNs
        pbar.set_description(f"Stage 0 — {steps[4]}")
        df["meter_reading"] = df["meter_reading"].clip(lower=0)
        df.dropna(subset=["meter_reading"], inplace=True)
        pbar.update(1)

        # Step 6 — Pivot to daily profiles
        pbar.set_description(f"Stage 0 — {steps[5]}")
        daily_profiles = df.pivot_table(
            index=["building_id", "site_id", "primary_use", "date"],
            columns="hour",
            values="meter_reading"
        )
        pbar.update(1)

        # Step 7 — Drop days with any missing hour
        pbar.set_description(f"Stage 0 — {steps[6]}")
        daily_profiles.dropna(inplace=True)
        CONSOLE.log(f"  {len(daily_profiles):,} complete daily profiles retained.")
        pbar.update(1)

        # Step 8 — Min-Max normalisation (per-hour column, across all buildings)
        pbar.set_description(f"Stage 0 — {steps[7]}")
        scaler = MinMaxScaler()
        normalised_values = scaler.fit_transform(daily_profiles.values.T).T
        normalised_profiles = pd.DataFrame(
            normalised_values,
            index=daily_profiles.index,
            columns=daily_profiles.columns
        )
        pbar.update(1)

        # Step 9 — Save
        pbar.set_description(f"Stage 0 — {steps[8]}")
        normalised_profiles.to_pickle(PROCESSED_DATA_PATH)
        pbar.update(1)

    CONSOLE.print(
        f"[bold green]Stage 0 Complete![/] Preprocessed data saved to "
        f"[cyan]{PROCESSED_DATA_PATH}[/cyan]."
    )


if __name__ == "__main__":
    run_preprocessing()
