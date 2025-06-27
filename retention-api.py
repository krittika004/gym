from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model_retain = joblib.load("models/retention_model.pkl")
scaler_retain = joblib.load("models/retention_scaler.pkl")

@app.route("/", methods=["GET"])
def home():
    return "✅ Retention Prediction API is running!"

@app.route("/predict/retention", methods=["POST"])
def predict_retention():
    data = request.get_json()

    # Extract features from input (no gender)
    features = np.array([
        data["age"],
        data["membership_duration"],
        data["visits_per_week"],
        data["trainer_assigned"]
    ]).reshape(1, -1)

    scaled = scaler_retain.transform(features)
    prediction = model_retain.predict(scaled)[0]

    return jsonify({"retained": int(prediction)})

if __name__ == "__main__":
    app.run(debug=False, port=5003, use_reloader=False)