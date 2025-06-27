from flask import Flask, request, jsonify
import math

app = Flask(__name__)

def predict_enrollment_gain(current_month: int, current_enrollment: int, marketing_campaign: int, months_to_predict: int):
    if not (1 <= current_month <= 12):
        raise ValueError("Month must be between 1 and 12")
    if not (1 <= months_to_predict <= 12):
        raise ValueError("Prediction range must be between 1 and 12 months")

    growth_rate = 1.05 if marketing_campaign == 1 else 1.02
    future_enrollment = current_enrollment * (growth_rate ** months_to_predict)
    gain = math.ceil(future_enrollment - current_enrollment)
    return gain

@app.route("/", methods=["GET"])
def home():
    return "✅ Enrollment API is running!"

@app.route("/predict/enrollment", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        current_month = data["current_month"]
        current_enrollment = data["current_enrollment"]
        marketing_campaign = data["marketing_campaign"]
        months_to_predict = data["months_to_predict"]

        gain = predict_enrollment_gain(current_month, current_enrollment, marketing_campaign, months_to_predict)

        return jsonify({
            "predicted_new_enrollment": gain
        })

    except KeyError as e:
        return jsonify({"error": f"Missing field: {e.args[0]}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, port=5001, use_reloader=False)
