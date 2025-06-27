from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model_maint = joblib.load("models/maintenance_model.pkl")
scaler_maint = joblib.load("models/maintenance_scaler.pkl")

@app.route("/", methods=["GET"])
def home():
    return "✅ Maintenance Prediction API is running!"

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

if __name__ == "__main__":
    app.run(debug=False, port=5002, use_reloader=False)
