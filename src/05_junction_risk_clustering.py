"""
05_junction_risk_clustering.py
Unsupervised ML: cluster the 15 junctions into risk segments using
KMeans on aggregated junction-level features (violation rate, accident
rate, avg speed-over-limit, density, waiting time, signal efficiency).
Also fits PCA for a 2D visualization.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

df = pd.read_csv("data/traffic_cleaned.csv", parse_dates=["Date"])

agg = df.groupby(["Junction_ID", "Junction_Name"]).agg(
    violation_rate=("Violation", lambda s: (s == "Yes").mean() * 100),
    accident_rate=("Accident", lambda s: (s == "Yes").mean() * 100),
    avg_speed_over=("Speed_Over_Limit", "mean"),
    avg_density=("Traffic_Density", "mean"),
    avg_waiting=("Waiting_Time", "mean"),
    avg_signal_eff=("Signal_Efficiency", "mean"),
).reset_index()
print(agg)

FEATURES = ["violation_rate", "accident_rate", "avg_speed_over", "avg_density", "avg_waiting", "avg_signal_eff"]
X = agg[FEATURES].values
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# --- choose k via silhouette score ---
best_k, best_score = 2, -1
scores = {}
for k in range(2, 6):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    score = silhouette_score(Xs, labels)
    scores[k] = score
    if score > best_score:
        best_k, best_score = k, score
print("Silhouette scores by k:", scores)
print(f"Chosen k={best_k} (silhouette={best_score:.3f})")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
agg["Cluster"] = kmeans.fit_predict(Xs)

# label clusters by mean risk (violation_rate + accident_rate) so labels are interpretable
cluster_risk = agg.groupby("Cluster")[["violation_rate", "accident_rate"]].mean().sum(axis=1).sort_values()
risk_labels = {c: lbl for c, lbl in zip(cluster_risk.index, ["Low Risk", "Medium Risk", "High Risk", "Critical"][:best_k])}
agg["Risk_Segment"] = agg["Cluster"].map(risk_labels)

print(agg[["Junction_Name", "violation_rate", "accident_rate", "Risk_Segment"]].sort_values("violation_rate", ascending=False))

agg.to_csv("reports/junction_risk_clusters.csv", index=False)
joblib.dump({"scaler": scaler, "kmeans": kmeans, "risk_labels": risk_labels, "features": FEATURES},
            "models/junction_risk_clustering.pkl")
print("Saved models/junction_risk_clustering.pkl")

# --- 2D PCA visualization ---
pca = PCA(n_components=2)
coords = pca.fit_transform(Xs)
agg["PC1"], agg["PC2"] = coords[:, 0], coords[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(list(scores.keys()), list(scores.values()), marker="o", color="#4C8DFF")
axes[0].set_title("Silhouette Score by k")
axes[0].set_xlabel("k (clusters)"); axes[0].set_ylabel("Silhouette score")
axes[0].axvline(best_k, color="#EF4A42", linestyle="--", alpha=0.6)

colors = {"Low Risk": "#34D9A0", "Medium Risk": "#F2A93B", "High Risk": "#EF4A42", "Critical": "#8B0000"}
for seg, group in agg.groupby("Risk_Segment"):
    axes[1].scatter(group["PC1"], group["PC2"], label=seg, s=90, color=colors.get(seg, "#4C8DFF"))
for _, row in agg.iterrows():
    axes[1].annotate(row["Junction_ID"], (row["PC1"], row["PC2"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
axes[1].set_title(f"Junction Risk Clusters (PCA, k={best_k})")
axes[1].legend()
plt.tight_layout()
plt.savefig("reports/figures/junction_clustering.png", dpi=140)
print("Saved reports/figures/junction_clustering.png")
