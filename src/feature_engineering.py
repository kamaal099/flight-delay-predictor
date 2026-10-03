import pandas as pd
import numpy as np


TARGET = "ArrDel15"


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create machine-learning features for flight delay prediction.
    """

    df = df.copy()

    # --------------------------------------------------
    # 1. Date features
    # --------------------------------------------------
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])

    df["Year"] = df["FlightDate"].dt.year
    df["Month"] = df["FlightDate"].dt.month
    df["Day"] = df["FlightDate"].dt.day
    df["DayOfWeek"] = df["FlightDate"].dt.dayofweek
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

    # --------------------------------------------------
    # 2. Departure time features
    # --------------------------------------------------
    # Convert HHMM → minutes after midnight
    df["DepHour"] = df["CRSDepTime"] // 100
    df["DepMinute"] = df["CRSDepTime"] % 100

    df["DepMinutes"] = (
        df["DepHour"] * 60 + df["DepMinute"]
    )

    # Time-of-day buckets
    df["DepTimeOfDay"] = pd.cut(
        df["DepHour"],
        bins=[-1, 5, 11, 16, 20, 24],
        labels=[
            "Night",
            "Morning",
            "Afternoon",
            "Evening",
            "LateNight"
        ]
    ).astype(str)

    # --------------------------------------------------
    # 3. Arrival scheduled time
    # --------------------------------------------------
    df["ArrHour"] = df["CRSArrTime"] // 100
    df["ArrMinute"] = df["CRSArrTime"] % 100

    # --------------------------------------------------
    # 4. Airline / route features
    # --------------------------------------------------
    df["Route"] = (
        df["Origin"].astype(str)
        + "_"
        + df["Dest"].astype(str)
    )

    # --------------------------------------------------
    # 5. Distance transformations
    # --------------------------------------------------
    df["DistanceLog"] = np.log1p(df["Distance"])

    # --------------------------------------------------
    # 6. Scheduled duration
    # --------------------------------------------------
    df["CRSElapsedTime"] = pd.to_numeric(
        df["CRSElapsedTime"],
        errors="coerce"
    )

    # --------------------------------------------------
    # 7. Remove columns that should NOT be used
    # --------------------------------------------------
    drop_columns = [
        "FlightDate",
        "Cancelled",
        "Diverted",
        "Tail_Number",
        "Flight_Number_Reporting_Airline",
        "Reporting_Airline"
    ]

    df = df.drop(
        columns=[
            col for col in drop_columns
            if col in df.columns
        ]
    )

    # --------------------------------------------------
    # 8. Handle missing values
    # --------------------------------------------------
    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns

    df[numeric_columns] = df[numeric_columns].fillna(
        df[numeric_columns].median()
    )

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns

    df[categorical_columns] = df[categorical_columns].fillna(
        "UNKNOWN"
    )

    return df