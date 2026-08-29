"""
02_eda.py
Generates exploratory data analysis plots saved to reports/figures/
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#0f1626", "figure.facecolor": "#0a0f1c",
                                      "grid.color": "#26314b", "text.color": "#dfe6f5",
                                      "axes.labelcolor": "#dfe6f5", "xtick.color": "#9aa7c2",
                                      "ytick.color": "#9aa7c2", "axes.edgecolor": "#26314b"})
PALETTE = ["#4C8DFF", "#EF4A42", "#F2A93B", "#34D9A0", "#8B7CF6", "#5B6787"]

df = pd.read_csv("data/traffic_cleaned.csv", parse_dates=["Date"])
print("Loaded:", df.shape)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# 1. Violation rate by vehicle type
vt = df.groupby("Vehicle_Type")["Violation"].apply(lambda s: (s == "Yes").mean() * 100).sort_values(ascending=False)
sns.barplot(x=vt.values, y=vt.index, ax=axes[0, 0], palette=PALETTE)
axes[0, 0].set_title("Violation Rate (%) by Vehicle Type", color="#dfe6f5")
axes[0, 0].set_xlabel("Violation Rate %")

# 2. Violations by hour
hr = df[df.Violation == "Yes"].groupby("Hour").size()
axes[0, 1].plot(hr.index, hr.values, color="#4C8DFF", linewidth=2)
axes[0, 1].fill_between(hr.index, hr.values, color="#4C8DFF", alpha=0.2)
axes[0, 1].set_title("Violations by Hour of Day", color="#dfe6f5")
axes[0, 1].set_xlabel("Hour")

# 3. Speed distribution by violation
sns.kdeplot(data=df, x="Vehicle_Speed", hue="Violation", ax=axes[1, 0],
            palette={"Yes": "#EF4A42", "No": "#34D9A0"}, fill=True, alpha=0.3)
axes[1, 0].set_title("Speed Distribution: Violation vs No Violation", color="#dfe6f5")

# 4. Correlation heatmap of numeric features
num_cols = ["Waiting_Time", "Vehicle_Speed", "Speed_Limit", "Speed_Over_Limit",
            "Traffic_Density", "Signal_Efficiency", "Fine_Amount"]
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 1], cbar=False, center=0)
axes[1, 1].set_title("Feature Correlation Matrix", color="#dfe6f5")

plt.tight_layout()
plt.savefig("reports/figures/eda_overview.png", dpi=140, facecolor="#0a0f1c")
print("Saved reports/figures/eda_overview.png")
