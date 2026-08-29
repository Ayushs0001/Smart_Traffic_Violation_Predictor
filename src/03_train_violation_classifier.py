"""
03_train_violation_classifier.py
Binary classification: will this vehicle observation be a Violation (Yes/No)?
Compares Logistic Regression, Random Forest, Gradient Boosting.
Saves the best model + evaluation plots.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, RocCurveDisplay, classification_report)

df = pd.read_csv("data/traffic_cleaned.csv", parse_dates=["Date"])

# --- Only use features that would be KNOWN at prediction time (avoid leakage:
#     drop Violation_Type, Fine_Amount, Stop_Line_Crossed etc. that are direct
#     consequences of / synonyms for the violation itself) ---
FEATURES_NUM = ["Hour", "Is_Peak_Hour", "Is_Weekend", "Vehicle_Speed", "Speed_Limit",
                 "Speed_Over_Limit", "Traffic_Density", "Waiting_Time", "Signal_Efficiency",
                 "Number_of_Lanes", "Lane"]
FEATURES_CAT = ["Vehicle_Type", "Signal_State", "Weather", "Road_Type", "Junction_Road_Condition",
                 "Direction", "Signal_Type"]
TARGET = "Violation"

X = df[FEATURES_NUM + FEATURES_CAT]
y = (df[TARGET] == "Yes").astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {X_train.shape}, Test: {X_test.shape}, Positive rate: {y.mean():.3f}")

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), FEATURES_NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
])

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5,
                                             class_weight="balanced", random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.08, random_state=42),
}

results = {}
fitted_pipelines = {}

for name, clf in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    results[name] = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    fitted_pipelines[name] = pipe
    print(f"\n{name}: {results[name]}")

results_df = pd.DataFrame(results).T.round(4)
results_df.to_csv("reports/violation_classifier_comparison.csv")
print("\n=== Model comparison ===")
print(results_df)

best_name = results_df["f1"].idxmax()
best_pipe = fitted_pipelines[best_name]
print(f"\nBest model by F1: {best_name}")

# --- save model ---
joblib.dump(best_pipe, "models/violation_classifier.pkl")
print("Saved models/violation_classifier.pkl")

# --- evaluation plots ---
pred = best_pipe.predict(X_test)
proba = best_pipe.predict_proba(X_test)[:, 1]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
plt.style.use("dark_background")

cm = confusion_matrix(y_test, pred)
im = axes[0].imshow(cm, cmap="Blues")
axes[0].set_title(f"Confusion Matrix — {best_name}")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["No Violation", "Violation"])
axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["No Violation", "Violation"])
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, cm[i, j], ha="center", va="center", color="white", fontsize=13)

RocCurveDisplay.from_predictions(y_test, proba, ax=axes[1], color="#4C8DFF")
axes[1].set_title("ROC Curve")
axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")

# feature importance (only for tree models)
if best_name in ("RandomForest", "GradientBoosting"):
    ohe = best_pipe.named_steps["prep"].named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(FEATURES_CAT)
    all_names = FEATURES_NUM + list(cat_names)
    importances = best_pipe.named_steps["clf"].feature_importances_
    top_idx = np.argsort(importances)[-12:]
    axes[2].barh([all_names[i] for i in top_idx], importances[top_idx], color="#EF4A42")
    axes[2].set_title("Top 12 Feature Importances")
else:
    coefs = best_pipe.named_steps["clf"].coef_[0]
    ohe = best_pipe.named_steps["prep"].named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(FEATURES_CAT)
    all_names = FEATURES_NUM + list(cat_names)
    top_idx = np.argsort(np.abs(coefs))[-12:]
    axes[2].barh([all_names[i] for i in top_idx], coefs[top_idx], color="#EF4A42")
    axes[2].set_title("Top 12 |Coefficients|")

plt.tight_layout()
plt.savefig("reports/figures/violation_classifier_eval.png", dpi=140, facecolor="#111")
print("Saved reports/figures/violation_classifier_eval.png")

with open("reports/violation_classifier_report.txt", "w") as f:
    f.write(f"Best model: {best_name}\n\n")
    f.write(classification_report(y_test, pred, target_names=["No Violation", "Violation"]))
