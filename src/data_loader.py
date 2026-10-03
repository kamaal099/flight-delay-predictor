import zipfile
from pathlib import Path

import pandas as pd
import numpy as np


REQUIRED_COLUMNS = [
    "FlightDate",
    "Reporting_Airline",
    "Tail_Number",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "Distance",
    "ArrDel15",
    "Cancelled",
    "Diverted",
]


def load_month(zip_path):
    zip_path = Path(zip_path)
    

    with zipfile.ZipFile(zip_path) as z:
        csv_files = [
            name for name in z.namelist()
            if name.lower().endswith(".csv")
        ]

        if len(csv_files) != 1:
            raise ValueError(
                f"Expected exactly one CSV in {zip_path.name}, "
                f"found: {csv_files}"
            )

        csv_name = csv_files[0]

        with z.open(csv_name) as csv_file:
            df = pd.read_csv(
                csv_file,
                usecols=REQUIRED_COLUMNS
            )

    df["FlightDate"] = pd.to_datetime(df["FlightDate"])

    return df
def load_months(paths):
    frames = []

    for path in paths:
        df = load_month(path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    return combined

def prepare_model_data(df):
    df = df[
        (df["Cancelled"] == 0) &
        (df["Diverted"] == 0)
    ].copy()

    df = df.drop_duplicates()

    return df