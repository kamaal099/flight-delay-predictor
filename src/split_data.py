import pandas as pd
from sklearn.model_selection import train_test_split


TARGET = "ArrDel15"


def split_data(df: pd.DataFrame):
    """
    Split the dataset into training and testing sets.

    Uses stratification so the delayed-flight ratio
    remains approximately equal in both sets.
    """

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test