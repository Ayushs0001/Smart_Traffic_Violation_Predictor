"""
app.py — Smart Traffic Signal & Violation Analytics
Interactive Streamlit dashboard: EDA + live ML predictions using the
saved scikit-learn pipelines (violation classifier, violation-type
classifier, waiting-time regressor, junction risk clustering).

Run with:  streamlit run app.py
"""
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG & THEME
# ============================================================
st.set_page_config(
    page_title="Smart Traffic Signal & Violation Analytics",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

C = {
    "bg": "#0A0F1C", "panel": "#111A2C", "border": "#212D45",
    "text": "#EAEEF7", "textDim": "#8894AD",
    "red": "#EF4A42", "amber": "#F2A93B", "green": "#34D9A0", "blue": "#4C8DFF", "violet": "#8B7CF6",
}

st.markdown(f"""
<style>
    .stApp {{ background-color: {C['bg']}; color: {C['text']}; }}
    section[data-testid="stSidebar"] {{ background-color: {C['panel']}; border-right: 1px solid {C['border']}; }}
    div[data-testid="stMetric"] {{
        background-color: {C['panel']}; border: 1px solid {C['border']}; border-radius: 10px;
        padding: 14px 16px;
    }}
    h1, h2, h3 {{ color: {C['text']}; font-family: 'Space Grotesk', sans-serif; }}
    .stButton>button {{
        background-color: {C['blue']}; color: white; border-radius: 8px; border: none; font-weight: 600;
    }}
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor=C["panel"], plot_bgcolor=C["panel"],
    font=dict(color=C["textDim"], family="Inter"),
    xaxis=dict(gridcolor=C["border"]), yaxis=dict(gridcolor=C["border"]),
    margin=dict(l=10, r=10, t=40, b=10),
)

# ============================================================
# DATA & MODEL LOADING (cached)
# ============================================================
@st.cache_data
def load_data():
    import os
    if not os.path.exists("data/traffic_cleaned.csv"):
        st.error("data/traffic_cleaned.csv not found. Run `python src/01_data_cleaning.py` first.")
        st.stop()
    df = pd.read_csv("data/traffic_cleaned.csv", parse_dates=["Date"])
    return df

@st.cache_resource
def load_models():
    return {
        "violation_clf": joblib.load("models/violation_classifier.pkl"),
        "vtype_bundle": joblib.load("models/violation_type_classifier.pkl"),
        "waiting_reg": joblib.load("models/waiting_time_regressor.pkl"),
        "cluster_bundle": joblib.load("models/junction_risk_clustering.pkl"),
    }

df = load_data()
models = load_models()

# ============================================================
# SIDEBAR NAV
# ============================================================
st.sidebar.markdown("### 🚦 Smart Traffic Analytics")
st.sidebar.caption("15 junctions · 6 months · 26,000 records")
page = st.sidebar.radio("Navigate", [
    "📊 Executive Overview",
    "🔍 Exploratory Analysis",
    "⚠️ Violation Risk Predictor",
    "🏷️ Violation Type Predictor",
    "⏱️ Waiting Time Predictor",
    "🗺️ Junction Risk Explorer",
], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.caption("Models: Random Forest (classification & regression), K-Means (clustering). Built with scikit-learn — no deep learning.")

# ============================================================
# PAGE 1 — EXECUTIVE OVERVIEW
# ============================================================
if page == "📊 Executive Overview":
    st.title("Executive Overview")
    st.caption("Feb 2026 – Jul 2026")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Vehicles", f"{len(df):,}")
    c2.metric("Violations", f"{(df.Violation=='Yes').sum():,}", f"{(df.Violation=='Yes').mean()*100:.1f}% rate")
    c3.metric("Fine Collected", f"₹{df.Fine_Amount.sum()/1e5:.1f}L")
    c4.metric("Avg Waiting Time", f"{df.Waiting_Time.mean():.1f}s")
    c5.metric("Accidents", f"{(df.Accident=='Yes').sum()}")
    c6.metric("Avg Speed", f"{df.Vehicle_Speed.mean():.1f} km/h")

    col1, col2 = st.columns([1.4, 1])
    with col1:
        by_junction = df[df.Violation == "Yes"].groupby("Junction_Name").size().sort_values(ascending=False).reset_index(name="Violations")
        fig = px.bar(by_junction, x="Violations", y="Junction_Name", orientation="h",
                      color="Violations", color_continuous_scale=[C["blue"], C["amber"], C["red"]])
        fig.update_layout(**PLOTLY_LAYOUT, title="Violations by Junction", height=420)
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        by_vehicle = df[df.Violation == "Yes"].groupby("Vehicle_Type").size().reset_index(name="Violations")
        fig = px.pie(by_vehicle, names="Vehicle_Type", values="Violations", hole=0.5,
                      color_discrete_sequence=[C["red"], C["blue"], C["amber"], C["violet"], C["green"], C["textDim"]])
        fig.update_layout(**PLOTLY_LAYOUT, title="Violations by Vehicle Type", height=420)
        st.plotly_chart(fig, use_container_width=True)

    monthly = df.groupby(df.Date.dt.strftime("%Y-%m")).agg(
        Total=("Record_ID", "count"), Violations=("Violation", lambda s: (s == "Yes").sum())
    ).reset_index().rename(columns={"Date": "Month"})
    fig = px.line(monthly, x="Month", y=["Total", "Violations"], markers=True,
                   color_discrete_sequence=[C["blue"], C["red"]])
    fig.update_layout(**PLOTLY_LAYOUT, title="Monthly Trend", height=320)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 2 — EDA
# ============================================================
elif page == "🔍 Exploratory Analysis":
    st.title("Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        hr = df[df.Violation == "Yes"].groupby("Hour").size().reset_index(name="Violations")
        fig = px.area(hr, x="Hour", y="Violations", color_discrete_sequence=[C["blue"]])
        fig.update_layout(**PLOTLY_LAYOUT, title="Violations by Hour of Day", height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        vt = df.groupby("Vehicle_Type")["Violation"].apply(lambda s: (s == "Yes").mean() * 100).sort_values(ascending=False).reset_index(name="Rate")
        fig = px.bar(vt, x="Rate", y="Vehicle_Type", orientation="h", color_discrete_sequence=[C["amber"]])
        fig.update_layout(**PLOTLY_LAYOUT, title="Violation Rate (%) by Vehicle Type", height=350)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.histogram(df, x="Vehicle_Speed", color="Violation", barmode="overlay", opacity=0.6,
                             color_discrete_map={"Yes": C["red"], "No": C["green"]})
        fig.update_layout(**PLOTLY_LAYOUT, title="Speed Distribution by Violation", height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        vtype = df[df.Violation == "Yes"].groupby("Violation_Type").size().sort_values(ascending=False).reset_index(name="Count")
        fig = px.bar(vtype, x="Count", y="Violation_Type", orientation="h", color_discrete_sequence=[C["violet"]])
        fig.update_layout(**PLOTLY_LAYOUT, title="Violation Type Breakdown", height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature correlation")
    num_cols = ["Waiting_Time", "Vehicle_Speed", "Speed_Limit", "Speed_Over_Limit", "Traffic_Density", "Signal_Efficiency", "Fine_Amount"]
    corr = df[num_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(**PLOTLY_LAYOUT, height=420)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 3 — VIOLATION RISK PREDICTOR
# ============================================================
elif page == "⚠️ Violation Risk Predictor":
    st.title("Violation Risk Predictor")
    st.caption("Binary classifier (Random Forest, ROC-AUC ≈ 0.72) — predicts risk from contextual features only, "
               "not from directly-observed rule flags (helmet/seatbelt/phone), to avoid a trivial task.")

    col1, col2, col3 = st.columns(3)
    with col1:
        hour = st.slider("Hour of day", 0, 23, 18)
        vehicle_type = st.selectbox("Vehicle type", ["Car", "Bike", "Auto", "Bus", "Truck", "Taxi"])
        weather = st.selectbox("Weather", ["Clear", "Cloudy", "Rain", "Fog"])
    with col2:
        speed = st.slider("Vehicle speed (km/h)", 0, 120, 78)
        speed_limit = st.selectbox("Speed limit (km/h)", [40, 50, 60, 80], index=2)
        road_type = st.selectbox("Road type", ["Residential", "City Road", "Main Road", "Highway"])
    with col3:
        density = st.slider("Traffic density", 0, 100, 82)
        signal_state = st.selectbox("Signal state", ["Red", "Yellow", "Green"])
        waiting_time = st.slider("Waiting time (s)", 0, 120, 45)

    is_peak = 1 if (8 <= hour <= 10 or 17 <= hour <= 20) else 0
    is_weekend = st.checkbox("Weekend", value=False)

    sample = pd.DataFrame([{
        "Hour": hour, "Is_Peak_Hour": is_peak, "Is_Weekend": int(is_weekend),
        "Vehicle_Speed": speed, "Speed_Limit": speed_limit, "Speed_Over_Limit": speed - speed_limit,
        "Traffic_Density": density, "Waiting_Time": waiting_time, "Signal_Efficiency": 0.5,
        "Number_of_Lanes": 4, "Lane": 2,
        "Vehicle_Type": vehicle_type, "Signal_State": signal_state, "Weather": weather,
        "Road_Type": road_type, "Junction_Road_Condition": "Average",
        "Direction": "North", "Signal_Type": "Automatic",
    }])

    if st.button("Predict violation risk", type="primary"):
        clf = models["violation_clf"]
        pred = clf.predict(sample)[0]
        proba = clf.predict_proba(sample)[0, 1]
        colA, colB = st.columns([1, 2])
        with colA:
            if pred == 1:
                st.error(f"⚠️ Predicted: VIOLATION LIKELY\n\nProbability: {proba:.1%}")
            else:
                st.success(f"✅ Predicted: No violation\n\nProbability of violation: {proba:.1%}")
        with colB:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=proba * 100,
                gauge={"axis": {"range": [0, 100]},
                        "bar": {"color": C["red"] if proba > 0.5 else C["green"]},
                        "steps": [{"range": [0, 40], "color": "rgba(52,217,160,0.25)"},
                                   {"range": [40, 70], "color": "rgba(242,169,59,0.25)"},
                                   {"range": [70, 100], "color": "rgba(239,74,66,0.25)"}]},
                title={"text": "Violation Probability (%)"},
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=250)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 4 — VIOLATION TYPE PREDICTOR
# ============================================================
elif page == "🏷️ Violation Type Predictor":
    st.title("Violation Type Predictor")
    st.caption("Multiclass classifier (Random Forest, Macro-F1 ≈ 0.37) — given a violation occurs, which type is most likely?")

    col1, col2 = st.columns(2)
    with col1:
        hour = st.slider("Hour of day", 0, 23, 18, key="t_hour")
        vehicle_type = st.selectbox("Vehicle type", ["Car", "Bike", "Auto", "Bus", "Truck", "Taxi"], key="t_vtype")
        weather = st.selectbox("Weather", ["Clear", "Cloudy", "Rain", "Fog"], key="t_weather")
        signal_state = st.selectbox("Signal state", ["Red", "Yellow", "Green"], key="t_signal")
    with col2:
        speed = st.slider("Vehicle speed (km/h)", 0, 120, 78, key="t_speed")
        speed_limit = st.selectbox("Speed limit (km/h)", [40, 50, 60, 80], index=2, key="t_limit")
        road_type = st.selectbox("Road type", ["Residential", "City Road", "Main Road", "Highway"], key="t_road")
        density = st.slider("Traffic density", 0, 100, 82, key="t_density")

    is_peak = 1 if (8 <= hour <= 10 or 17 <= hour <= 20) else 0

    sample = pd.DataFrame([{
        "Hour": hour, "Is_Peak_Hour": is_peak, "Is_Weekend": 0,
        "Vehicle_Speed": speed, "Speed_Limit": speed_limit, "Speed_Over_Limit": speed - speed_limit,
        "Traffic_Density": density, "Waiting_Time": 30,
        "Vehicle_Type": vehicle_type, "Signal_State": signal_state, "Weather": weather, "Road_Type": road_type,
    }])

    if st.button("Predict violation type", type="primary"):
        bundle = models["vtype_bundle"]
        pipe, le = bundle["pipeline"], bundle["label_encoder"]
        proba = pipe.predict_proba(sample)[0]
        top_idx = np.argsort(proba)[::-1]
        result = pd.DataFrame({"Violation_Type": le.classes_[top_idx], "Probability": proba[top_idx]})
        st.subheader(f"Most likely: {result.iloc[0]['Violation_Type']} ({result.iloc[0]['Probability']:.1%})")
        fig = px.bar(result, x="Probability", y="Violation_Type", orientation="h",
                      color="Probability", color_continuous_scale=[C["blue"], C["amber"], C["red"]])
        fig.update_layout(**PLOTLY_LAYOUT, height=380)
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 5 — WAITING TIME PREDICTOR
# ============================================================
elif page == "⏱️ Waiting Time Predictor":
    st.title("Waiting Time Predictor")
    st.caption("Regression model (Random Forest, R² ≈ 0.95, MAE ≈ 4.4s) — simulate signal retiming scenarios.")

    col1, col2 = st.columns(2)
    with col1:
        red = st.slider("Red duration (s)", 20, 100, 80)
        yellow = st.slider("Yellow duration (s)", 3, 6, 4)
        green = st.slider("Green duration (s)", 20, 100, 60)
        signal_state = st.selectbox("Current signal state", ["Red", "Yellow", "Green"], key="w_signal")
    with col2:
        density = st.slider("Traffic density", 0, 100, 82, key="w_density")
        hour = st.slider("Hour of day", 0, 23, 18, key="w_hour")
        road_type = st.selectbox("Road type", ["Residential", "City Road", "Main Road", "Highway"], key="w_road")
        weather = st.selectbox("Weather", ["Clear", "Cloudy", "Rain", "Fog"], key="w_weather")

    cycle = red + yellow + green
    is_peak = 1 if (8 <= hour <= 10 or 17 <= hour <= 20) else 0
    efficiency = green / cycle

    sample = pd.DataFrame([{
        "Traffic_Density": density, "Red_Duration": red, "Yellow_Duration": yellow, "Green_Duration": green,
        "Cycle_Time": cycle, "Signal_Efficiency": efficiency, "Hour": hour, "Is_Peak_Hour": is_peak,
        "Signal_State": signal_state, "Road_Type": road_type, "Weather": weather,
    }])

    if st.button("Predict waiting time", type="primary"):
        pred = models["waiting_reg"].predict(sample)[0]
        st.metric("Predicted waiting time", f"{pred:.1f} seconds")
        st.caption(f"Cycle time: {cycle}s · Signal efficiency: {efficiency:.1%}")

# ============================================================
# PAGE 6 — JUNCTION RISK EXPLORER
# ============================================================
elif page == "🗺️ Junction Risk Explorer":
    st.title("Junction Risk Explorer")
    st.caption("K-Means clustering (unsupervised) on violation rate, accident rate, density, waiting time and signal efficiency.")

    agg = df.groupby(["Junction_ID", "Junction_Name"]).agg(
        violation_rate=("Violation", lambda s: (s == "Yes").mean() * 100),
        accident_rate=("Accident", lambda s: (s == "Yes").mean() * 100),
        avg_speed_over=("Speed_Over_Limit", "mean"),
        avg_density=("Traffic_Density", "mean"),
        avg_waiting=("Waiting_Time", "mean"),
        avg_signal_eff=("Signal_Efficiency", "mean"),
    ).reset_index()

    bundle = models["cluster_bundle"]
    scaler, kmeans, risk_labels, feats = bundle["scaler"], bundle["kmeans"], bundle["risk_labels"], bundle["features"]
    Xs = scaler.transform(agg[feats].values)
    agg["Cluster"] = kmeans.predict(Xs)
    agg["Risk_Segment"] = agg["Cluster"].map(risk_labels)

    color_map = {"Low Risk": C["green"], "Medium Risk": C["amber"], "High Risk": C["red"], "Critical": "#8B0000"}
    fig = px.scatter(agg, x="violation_rate", y="accident_rate", size="avg_density", color="Risk_Segment",
                       hover_name="Junction_Name", color_discrete_map=color_map, size_max=30)
    fig.update_layout(**PLOTLY_LAYOUT, title="Risk Matrix: Violation Rate vs Accident Rate", height=450,
                        xaxis_title="Violation Rate (%)", yaxis_title="Accident Rate (%)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Junction risk table")
    display = agg[["Junction_Name", "violation_rate", "accident_rate", "avg_waiting", "avg_signal_eff", "Risk_Segment"]].copy()
    display.columns = ["Junction", "Violation Rate (%)", "Accident Rate (%)", "Avg Waiting (s)", "Signal Efficiency", "Risk Segment"]
    display = display.sort_values("Violation Rate (%)", ascending=False).round(2)
    st.dataframe(display, use_container_width=True, hide_index=True)
