"""
01_data_cleaning.py
Cleans the raw (intentionally messy) Traffic_Facts.csv and merges
in the dimension-table CSVs to produce a single curated dataset for ML.

Reads from data/*.csv (not Excel) -- simpler, no openpyxl dependency,
works everywhere.
"""
import pandas as pd
import numpy as np
import re
import os

DATA_DIR = "data"
OUT_PATH = os.path.join(DATA_DIR, "traffic_cleaned.csv")


def clean_vehicle_type(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().lower()
    mapping = {
        "bike": "Bike", "two wheeler": "Bike", "two-wheeler": "Bike",
        "car": "Car", "truck": "Truck", "auto": "Auto", "auto rickshaw": "Auto",
        "bus": "Bus", "taxi": "Taxi", "cab": "Taxi",
    }
    return mapping.get(x, x.title())


def parse_mixed_date(x):
    if pd.isna(x):
        return pd.NaT
    x = str(x).strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d %b %Y"):
        try:
            return pd.to_datetime(x, format=fmt)
        except ValueError:
            continue
    return pd.to_datetime(x, errors="coerce")


def parse_mixed_time(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    ampm = bool(re.search(r"[APap][Mm]", x))
    fmt = "%I:%M:%S %p" if ampm else "%H:%M:%S"
    try:
        t = pd.to_datetime(x, format=fmt).time()
        return t.hour * 3600 + t.minute * 60 + t.second
    except ValueError:
        return np.nan


def main():
    print("Loading raw CSV files...")
    facts = pd.read_csv(os.path.join(DATA_DIR, "Traffic_Facts.csv"))
    junction = pd.read_csv(os.path.join(DATA_DIR, "Junction_Master.csv"))
    signal = pd.read_csv(os.path.join(DATA_DIR, "Signal_Master.csv"))
    print(f"Raw Traffic_Facts: {facts.shape}")

    # --- text cleanup (preserve real nulls, don't touch the literal "None"
    #     category used for Violation_Type when no violation occurred) ---
    text_cols = facts.select_dtypes(include="object").columns
    for c in text_cols:
        was_null = facts[c].isna()
        facts[c] = facts[c].astype(str).str.strip()
        facts.loc[was_null, c] = np.nan
        facts[c] = facts[c].replace({"nan": np.nan})

    facts["Junction_ID"] = facts["Junction_ID"].str.upper()
    facts["Vehicle_Type"] = facts["Vehicle_Type"].apply(clean_vehicle_type)

    # --- dates & time ---
    facts["Date"] = facts["Date"].apply(parse_mixed_date)
    facts["Time_Seconds"] = facts["Time"].apply(parse_mixed_time)
    facts["Hour"] = (facts["Time_Seconds"] // 3600).astype("Int64")

    # --- duplicates ---
    before = len(facts)
    facts = facts.drop_duplicates(subset="Record_ID", keep="first")
    print(f"Removed {before - len(facts)} duplicate rows")

    # --- nulls: fill categorical unknowns explicitly, don't silently drop ---
    for c in ["Helmet", "Seat_Belt", "Weather", "Road_Condition"]:
        facts[c] = facts[c].fillna("Unknown")
    facts["Lane"] = facts["Lane"].fillna(facts["Lane"].median())
    # Violation_Type is blank when Violation == 'No' (no violation occurred).
    # This is expected, not a data-quality issue -- encode explicitly.
    facts["Violation_Type"] = facts["Violation_Type"].fillna("None")

    # --- merge dimensions ---
    facts = facts.merge(
        junction[["Junction_ID", "Junction_Name", "Road_Type", "Number_of_Lanes",
                  "Signal_Type", "Speed_Limit", "Road_Condition"]]
        .rename(columns={"Speed_Limit": "Junction_Speed_Limit", "Road_Condition": "Junction_Road_Condition"}),
        on="Junction_ID", how="left"
    )
    facts = facts.merge(
        signal[["Junction_ID", "Red_Duration", "Yellow_Duration", "Green_Duration", "Cycle_Time"]],
        on="Junction_ID", how="left"
    )

    # --- derived features useful downstream ---
    facts["Is_Peak_Hour"] = facts["Hour"].apply(lambda h: 1 if (8 <= h <= 10 or 17 <= h <= 20) else 0)
    facts["Day_Of_Week"] = facts["Date"].dt.day_name()
    facts["Is_Weekend"] = facts["Date"].dt.dayofweek.isin([5, 6]).astype(int)
    facts["Speed_Over_Limit"] = facts["Vehicle_Speed"] - facts["Speed_Limit"]
    facts["Signal_Efficiency"] = facts["Green_Duration"] / facts["Cycle_Time"]

    facts.to_csv(OUT_PATH, index=False)
    print(f"Saved cleaned dataset: {OUT_PATH}  shape={facts.shape}")
    nulls = facts.isna().sum()
    print(nulls[nulls > 0] if (nulls > 0).any() else "No remaining nulls.")


if __name__ == "__main__":
    main()
