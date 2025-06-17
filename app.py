from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model_cls = joblib.load("model_klasifikasi.joblib")
model_reg = joblib.load("model_regresi.joblib")
label_encoder = joblib.load("label_encoder.joblib")

@app.route("/")
def home():
    return "Smart Farming ML Backend aktif!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    input_features = [
        data['soil_moisture'],
        data['soil_N'],
        data['soil_P'],
        data['soil_K'],
        data['hours_until_rain'],
        data['temperature']
    ]

    input_array = np.array(input_features).reshape(1, -1)
    pred_action_encoded = model_cls.predict(input_array)[0]
    pred_action = label_encoder.inverse_transform([pred_action_encoded])[0]

    # Default output
    hasil = {
        "status": pred_action,
        "air": 0.0,
        "nutrisi": 0.0
    }

    if pred_action != "tidak_ada_aksi":
        reg_pred = model_reg.predict(input_array)[0]
        air = reg_pred[0]
        nutrisi_total = reg_pred[1] + reg_pred[2] + reg_pred[3]

        hasil["air"] = round(float(air), 2)
        hasil["nutrisi"] = round(float(nutrisi_total), 2)

    return jsonify(hasil)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
