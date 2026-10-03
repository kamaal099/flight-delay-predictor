import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)


from src.data_loader import load_months
from src.feature_engineering import create_features
from src.split_data import split_data
from src.prepare_model_data import prepare_for_lightgbm


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading data...")

df = load_months([
    "data/raw/june.zip",
    "data/raw/july.zip"
])

# Remove cancelled and diverted flights
df = df[
    (df["Cancelled"] == 0) &
    (df["Diverted"] == 0)
].copy()

print("Dataset:", df.shape)


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

df = create_features(df)


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = split_data(df)

X_train, X_test = prepare_for_lightgbm(
    X_train,
    X_test
)


# ============================================================
# 4. LOAD TRAINED LIGHTGBM MODEL
# ============================================================

print("\nLoading model...")

model = joblib.load(
    "models/flight_delay_lgbm.pkl"
)

print("Model loaded successfully.")


# ============================================================
# 5. PREDICT PROBABILITIES
# ============================================================

print("\nGenerating predictions...")

y_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# 6. BASELINE PERFORMANCE - THRESHOLD 0.50
# ============================================================

baseline_threshold = 0.50

y_pred_baseline = (
    y_prob >= baseline_threshold
).astype(int)


baseline_precision = precision_score(
    y_test,
    y_pred_baseline,
    zero_division=0
)

baseline_recall = recall_score(
    y_test,
    y_pred_baseline,
    zero_division=0
)

baseline_f1 = f1_score(
    y_test,
    y_pred_baseline,
    zero_division=0
)


print("\n================================")
print("BASELINE PERFORMANCE")
print("================================")

print(f"Threshold: {baseline_threshold:.2f}")
print(f"Precision: {baseline_precision:.4f}")
print(f"Recall:    {baseline_recall:.4f}")
print(f"F1 Score:  {baseline_f1:.4f}")


# ============================================================
# 7. ROC-AUC
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    y_prob
)


# ============================================================
# 8. PR-AUC
# ============================================================

pr_auc = average_precision_score(
    y_test,
    y_prob
)


print("\n================================")
print("OVERALL MODEL PERFORMANCE")
print("================================")

print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC:  {pr_auc:.4f}")


# ============================================================
# 9. FIND BETTER CLASSIFICATION THRESHOLD
# ============================================================

print("\n================================")
print("THRESHOLD OPTIMIZATION")
print("================================")

# We want to catch at least 80% of delayed flights.
TARGET_RECALL = 0.80

thresholds = [
    round(x, 2)
    for x in [
        0.20 + i * 0.02
        for i in range(21)
    ]
]

threshold_results = []


for threshold in thresholds:

    y_pred_threshold = (
        y_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })


threshold_df = pd.DataFrame(
    threshold_results
)


print("\nThreshold results:")

print(
    threshold_df.to_string(
        index=False,
        formatters={
            "Threshold": "{:.2f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1": "{:.4f}".format
        }
    )
)


# ============================================================
# 10. SELECT BEST THRESHOLD
# ============================================================

valid_thresholds = threshold_df[
    threshold_df["Recall"] >= TARGET_RECALL
]


if len(valid_thresholds) > 0:

    # Among thresholds achieving target recall,
    # choose the one with highest precision.

    best_row = valid_thresholds.loc[
        valid_thresholds["Precision"].idxmax()
    ]

else:

    # If 80% recall cannot be achieved,
    # select threshold with highest recall.

    best_row = threshold_df.loc[
        threshold_df["Recall"].idxmax()
    ]

    print(
        "\nWARNING: No threshold achieved "
        f"{TARGET_RECALL:.0%} recall."
    )


best_threshold = float(
    best_row["Threshold"]
)


print("\n================================")
print("SELECTED THRESHOLD")
print("================================")

print(
    f"Selected threshold: {best_threshold:.2f}"
)

print(
    f"Expected delay recall: "
    f"{best_row['Recall']:.4f}"
)

print(
    f"Expected precision: "
    f"{best_row['Precision']:.4f}"
)

print(
    f"Expected F1: "
    f"{best_row['F1']:.4f}"
)


# ============================================================
# 11. FINAL PREDICTIONS USING OPTIMIZED THRESHOLD
# ============================================================

y_pred = (
    y_prob >= best_threshold
).astype(int)


precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# ============================================================
# 12. FINAL MODEL PERFORMANCE
# ============================================================

print("\n================================")
print("FINAL MODEL PERFORMANCE")
print("================================")

print(f"ROC-AUC:       {roc_auc:.4f}")
print(f"PR-AUC:        {pr_auc:.4f}")
print(f"Threshold:     {best_threshold:.2f}")
print(f"Precision:     {precision:.4f}")
print(f"Delay Recall:  {recall:.4f}")
print(f"F1 Score:      {f1:.4f}")


# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

print("\n================================")
print("CLASSIFICATION REPORT")
print("================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "On Time",
            "Delayed"
        ],
        zero_division=0
    )
)

# ============================================================
# PRECISION / RECALL VS THRESHOLD
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["F1"],
    marker="o",
    label="F1 Score"
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Selected Threshold ({best_threshold:.2f})"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Score")
plt.title("Precision, Recall and F1 vs Classification Threshold")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "models/threshold_analysis.png",
    dpi=200
)

plt.show()


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n================================")
print("CONFUSION MATRIX")
print("================================")

print(cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "On Time",
        "Delayed"
    ]
)

disp.plot()

plt.title(
    f"Flight Delay Prediction - "
    f"Optimized Threshold ({best_threshold:.2f})"
)

plt.tight_layout()

plt.savefig(
    "models/confusion_matrix.png",
    dpi=200
)

plt.show()


# ============================================================
# 15. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)


print("\n================================")
print("TOP 15 FEATURES")
print("================================")

print(
    importance.head(15).to_string(
        index=False
    )
)


# ============================================================
# 16. FEATURE IMPORTANCE PLOT
# ============================================================

top_features = (
    importance
    .head(15)
    .sort_values("Importance")
)


plt.figure(
    figsize=(10, 7)
)

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title(
    "Top 15 Features - LightGBM"
)

plt.tight_layout()

plt.savefig(
    "models/feature_importance.png",
    dpi=200
)

plt.show()


# ============================================================
# 17. SAVE EVALUATION RESULTS
# ============================================================

results = pd.DataFrame({
    "Metric": [
        "ROC-AUC",
        "PR-AUC",
        "Baseline Threshold",
        "Optimized Threshold",
        "Baseline Precision",
        "Baseline Recall",
        "Baseline F1",
        "Optimized Precision",
        "Optimized Recall",
        "Optimized F1"
    ],

    "Value": [
        roc_auc,
        pr_auc,
        baseline_threshold,
        best_threshold,
        baseline_precision,
        baseline_recall,
        baseline_f1,
        precision,
        recall,
        f1
    ]
})


results.to_csv(
    "models/evaluation_results.csv",
    index=False
)


# ============================================================
# 18. SAVE THRESHOLD FOR FUTURE API / DASHBOARD
# ============================================================

threshold_config = {
    "model": "LightGBM",
    "threshold": best_threshold,
    "target_recall": TARGET_RECALL,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc
}


with open(
    "models/threshold.json",
    "w"
) as f:

    json.dump(
        threshold_config,
        f,
        indent=4
    )


# ============================================================
# 19. SAVE THRESHOLD TEST RESULTS
# ============================================================

threshold_df.to_csv(
    "models/threshold_results.csv",
    index=False
)


print("\n================================")
print("FILES SAVED")
print("================================")

print("✓ models/confusion_matrix.png")
print("✓ models/feature_importance.png")
print("✓ models/evaluation_results.csv")
print("✓ models/threshold.json")
print("✓ models/threshold_results.csv")

print("\nEvaluation complete.")