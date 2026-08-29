"""
predict_demo.py
Demonstrates loading the saved models and running predictions on new
(unseen) vehicle observations. This is what a real-time scoring
service would call.
"""
import pandas as pd
import joblib

violation_clf = joblib.load("models/violation_classifier.pkl")
vtype_bundle = joblib.load("models/violation_type_classifier.pkl")
vtype_clf, vtype_le = vtype_bundle["pipeline"], vtype_bundle["label_encoder"]
waiting_reg = joblib.load("models/waiting_time_regressor.pkl")

# ---- Example 1: will this vehicle violate a rule? ----
sample = pd.DataFrame([{
    "Hour": 18, "Is_Peak_Hour": 1, "Is_Weekend": 0,
    "Vehicle_Speed": 78, "Speed_Limit": 60, "Speed_Over_Limit": 18,
    "Traffic_Density": 82.0, "Waiting_Time": 45.0, "Signal_Efficiency": 0.42,
    "Number_of_Lanes": 4, "Lane": 2,
    "Vehicle_Type": "Car", "Signal_State": "Red", "Weather": "Clear",
    "Road_Type": "Main Road", "Junction_Road_Condition": "Average",
    "Direction": "North", "Signal_Type": "Automatic",
}])

pred = violation_clf.predict(sample)[0]
proba = violation_clf.predict_proba(sample)[0, 1]
print("=== Violation prediction ===")
print(f"Prediction: {'VIOLATION' if pred == 1 else 'No violation'}  (probability = {proba:.2%})")

# ---- Example 2: given a violation, which type is most likely? ----
sample2 = pd.DataFrame([{
    "Hour": 18, "Is_Peak_Hour": 1, "Is_Weekend": 0,
    "Vehicle_Speed": 78, "Speed_Limit": 60, "Speed_Over_Limit": 18,
    "Traffic_Density": 82.0, "Waiting_Time": 45.0,
    "Vehicle_Type": "Car", "Signal_State": "Red", "Weather": "Clear", "Road_Type": "Main Road",
}])
type_pred = vtype_clf.predict(sample2)
type_label = vtype_le.inverse_transform(type_pred)[0]
print("\n=== Violation type prediction ===")
print(f"Most likely violation type: {type_label}")

# ---- Example 3: predicted signal waiting time ----
sample3 = pd.DataFrame([{
    "Traffic_Density": 82.0, "Red_Duration": 80, "Yellow_Duration": 4, "Green_Duration": 60,
    "Cycle_Time": 144, "Signal_Efficiency": 60 / 144, "Hour": 18, "Is_Peak_Hour": 1,
    "Signal_State": "Red", "Road_Type": "Main Road", "Weather": "Clear",
}])
wait_pred = waiting_reg.predict(sample3)[0]
print("\n=== Waiting time prediction ===")
print(f"Predicted waiting time: {wait_pred:.1f} seconds")
