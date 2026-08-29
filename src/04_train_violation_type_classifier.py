"""
04_train_violation_type_classifier.py
Multiclass classification: GIVEN a violation occurred, which type is it?
(Over Speeding / No Helmet / No Seat Belt / Wrong Lane / etc.)
Useful for prioritizing enforcement resource type per context.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

df = pd.read_csv("data/traffic_cleaned.csv", parse_dates=["Date"])
df = df[df["Violation"] == "Yes"].copy()
print("Violation-only subset:", df.shape)

FEATURES_NUM = ["Hour", "Is_Peak_Hour", "Is_Weekend", "Vehicle_Speed", "Speed_Limit",
                 "Speed_Over_Limit", "Traffic_Density", "Waiting_Time"]
FEATURES_CAT = ["Vehicle_Type", "Signal_State", "Weather", "Road_Type"]
TARGET = "Violation_Type"

X = df[FEATURES_NUM + FEATURES_CAT]
le = LabelEncoder()
y = le.fit_transform(df[TARGET])
print("Classes:", list(le.classes_))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), FEATURES_NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
])

clf = RandomForestClassifier(n_estimators=300, max_depth=14, min_samples_leaf=3,
                               class_weight="balanced", random_state=42, n_jobs=-1)
pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average="macro")
print(f"Accuracy: {acc:.4f}  Macro-F1: {f1:.4f}")
print(classification_report(y_test, pred, target_names=le.classes_))

joblib.dump({"pipeline": pipe, "label_encoder": le}, "models/violation_type_classifier.pkl")
print("Saved models/violation_type_classifier.pkl")

with open("reports/violation_type_classifier_report.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\nMacro-F1: {f1:.4f}\n\n")
    f.write(classification_report(y_test, pred, target_names=le.classes_))

# --- confusion matrix plot ---
cm = confusion_matrix(y_test, pred)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(le.classes_))); ax.set_xticklabels(le.classes_, rotation=60, ha="right")
ax.set_yticks(range(len(le.classes_))); ax.set_yticklabels(le.classes_)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Violation Type — Confusion Matrix (Macro-F1={f1:.2f})")
for i in range(len(le.classes_)):
    for j in range(len(le.classes_)):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
plt.tight_layout()
plt.savefig("reports/figures/violation_type_confusion_matrix.png", dpi=140)
print("Saved reports/figures/violation_type_confusion_matrix.png")
