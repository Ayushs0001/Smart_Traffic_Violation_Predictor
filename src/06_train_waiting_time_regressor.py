"""
06_train_waiting_time_regressor.py
Bonus regression model: predict signal Waiting_Time (seconds) from
junction/signal/traffic context. Useful for "what-if" signal retiming.
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

df = pd.read_csv("data/traffic_cleaned.csv", parse_dates=["Date"])

FEATURES_NUM = ["Traffic_Density", "Red_Duration", "Yellow_Duration", "Green_Duration",
                 "Cycle_Time", "Signal_Efficiency", "Hour", "Is_Peak_Hour"]
FEATURES_CAT = ["Signal_State", "Road_Type", "Weather"]
TARGET = "Waiting_Time"

X = df[FEATURES_NUM + FEATURES_CAT]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), FEATURES_NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
])

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=250, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1),
}

best_name, best_r2, best_pipe = None, -np.inf, None
rows = []
for name, reg in models.items():
    pipe = Pipeline([("prep", preprocessor), ("reg", reg)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)
    rows.append({"model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 3)})
    print(name, rows[-1])
    if r2 > best_r2:
        best_name, best_r2, best_pipe = name, r2, pipe

pd.DataFrame(rows).to_csv("reports/waiting_time_regressor_comparison.csv", index=False)
joblib.dump(best_pipe, "models/waiting_time_regressor.pkl")
print(f"Best: {best_name} (R2={best_r2:.3f}) saved to models/waiting_time_regressor.pkl")

pred = best_pipe.predict(X_test)
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, pred, alpha=0.15, s=8, color="#4C8DFF")
lims = [0, max(y_test.max(), pred.max())]
ax.plot(lims, lims, "--", color="#EF4A42")
ax.set_xlabel("Actual Waiting Time (s)")
ax.set_ylabel("Predicted Waiting Time (s)")
ax.set_title(f"{best_name} — Actual vs Predicted (R²={best_r2:.2f})")
plt.tight_layout()
plt.savefig("reports/figures/waiting_time_regression.png", dpi=140)
print("Saved reports/figures/waiting_time_regression.png")
