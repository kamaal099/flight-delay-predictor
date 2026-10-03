import joblib
import lightgbm as lgb

from src.data_loader import load_months
from src.feature_engineering import create_features
from src.split_data import split_data
from src.prepare_model_data import prepare_for_lightgbm


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

df = load_months([
    "data/raw/june.zip",
    "data/raw/july.zip"
])

# Remove cancelled/diverted flights
df = df[
    (df["Cancelled"] == 0) &
    (df["Diverted"] == 0)
].copy()

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. Feature engineering
# --------------------------------------------------

df = create_features(df)

print("Feature-engineered shape:", df.shape)


# --------------------------------------------------
# 3. Train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = split_data(df)

print("Training:", X_train.shape)
print("Testing:", X_test.shape)


# --------------------------------------------------
# 4. Prepare categorical variables
# --------------------------------------------------

X_train, X_test = prepare_for_lightgbm(
    X_train,
    X_test
)


# --------------------------------------------------
# 5. Create LightGBM model
# --------------------------------------------------

model = lgb.LGBMClassifier(
    objective="binary",

    n_estimators=1000,

    learning_rate=0.05,

    num_leaves=63,

    max_depth=-1,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42,

    n_jobs=-1,

    class_weight="balanced"
)


# --------------------------------------------------
# 6. Train
# --------------------------------------------------

print("\nStarting LightGBM training...\n")

model.fit(
    X_train,
    y_train,

    eval_set=[
        (X_train, y_train),
        (X_test, y_test)
    ],

    eval_metric="auc",

    categorical_feature=[
        col for col in [
            "Origin",
            "Dest",
            "Route",
            "DepTimeOfDay"
        ]
        if col in X_train.columns
    ],

    callbacks=[
        lgb.early_stopping(
            stopping_rounds=50
        ),
        lgb.log_evaluation(
            period=50
        )
    ]
)


# --------------------------------------------------
# 7. Save model
# --------------------------------------------------

joblib.dump(
    model,
    "models/flight_delay_lgbm.pkl"
)

print("\nModel saved successfully.")

print(
    "Best iteration:",
    model.best_iteration_
)

print(
    "Best validation AUC:",
    model.best_score_["valid_1"]["auc"]
)