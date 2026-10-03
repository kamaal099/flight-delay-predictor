import pandas as pd


CATEGORICAL_COLUMNS = [
    "Origin",
    "Dest",
    "Route",
    "DepTimeOfDay"
]


def prepare_for_lightgbm(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
):
    """
    Prepare categorical features for LightGBM.
    """

    X_train = X_train.copy()
    X_test = X_test.copy()

    for col in CATEGORICAL_COLUMNS:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")

            # Ensure test categories match training categories
            X_test[col] = pd.Categorical(
                X_test[col],
                categories=X_train[col].cat.categories
            )

    return X_train, X_test