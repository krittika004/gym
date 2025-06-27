from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from flask_cors import CORS

# === Load Models and Scalers ===
model_enroll = joblib.load("./models/enrollment_model.pkl")
scaler_enroll = joblib.load("./models/enrollment_scaler.pkl")

model_maint = joblib.load("./models/maintenance_model.pkl")
scaler_maint = joblib.load("./models/maintenance_scaler.pkl")

model_retain = joblib.load("./models/retention_model.pkl")
scaler_retain = joblib.load("./models/retention_scaler.pkl")

def predict_enrollment_gain(current_month: int, current_enrollment: int, marketing_campaign: int, months_to_predict: int):
    if not (1 <= current_month <= 12):
        raise ValueError("Month must be between 1 and 12")
    if not (1 <= months_to_predict <= 12):
        raise ValueError("Prediction range must be between 1 and 12 months")

    growth_rate = 1.05 if marketing_campaign == 1 else 1.02
    future_enrollment = current_enrollment * (growth_rate ** months_to_predict)
    gain = math.ceil(future_enrollment - current_enrollment)
    return gain

# === Initialize Flask App ===
app = Flask(__name__)
CORS(app)

# === Home Route ===
@app.route("/", methods=["GET"])
def home():
    return "✅ Gym Intelligence API is running!"

# === Enrollment Prediction ===
@app.route("/predict/enrollment", methods=["POST"])
def predict():
    data = request.get_json()
    current_month = data["current_month"]
    current_enrollment = data["current_enrollment"]
    marketing_campaign = data["marketing_campaign"]
    months_to_predict = data["months_to_predict"]
    gain = predict_enrollment_gain(current_month, current_enrollment, marketing_campaign, months_to_predict)

    return jsonify({"predicted_new_enrollment": gain})

'''@app.route("/predict/enrollment", methods=["POST"])
def predict_enrollment():
    data = request.get_json()
    features = np.array([
        data["month"],
        data["marketing_campaign"],
        data["enrollments_lag1"]
    ]).reshape(1, -1)
    scaled = scaler_enroll.transform(features)
    prediction = model_enroll.predict(scaled)[0]
    return jsonify({"predicted_enrollment": round(float(prediction), 2)})'''

# === Maintenance Prediction ===
@app.route("/predict/maintenance", methods=["POST"])
def predict_maintenance():
    data = request.get_json()
    features = np.array([
        data["running_hours"],
        data["load"],
        data["age"],
        data["days_since_last_maintenance"]
    ]).reshape(1, -1)
    scaled = scaler_maint.transform(features)
    prediction = model_maint.predict(scaled)[0]
    return jsonify({"needs_maintenance": int(prediction)})

# === Retention Prediction ===
@app.route("/predict/retention", methods=["POST"])
def predict_retention():
    data = request.get_json()
    features = np.array([
        data["age"],
        data["membership_duration"],
        data["visits_per_week"],
        data["trainer_assigned"]
    ]).reshape(1, -1)
    scaled = scaler_retain.transform(features)
    prediction = model_retain.predict(scaled)[0]
    return jsonify({"retained": int(prediction)})

# === Run App ===
if __name__ == "__main__":
    app.run(debug=False, port=5000, use_reloader=False)
