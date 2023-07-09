from flask import Flask, request, jsonify
import numpy as np
from sklearn.externals import joblib

app = Flask(__name__)

# Load the pre-trained scikit-learn model
model = {
    "replay-attack": joblib.load("./models/replay-attack_ycrcb_luv_extraTreesClassifier.pkl"),
    "print-attack": joblib.load("./models/print-attack_ycrcb_luv_extraTreesClassifier.pkl")
}

@app.route("/predict/replay", methods=["POST"])
def predict_replay():
    try:
        # Get the input data as a NumPy array from the request
        input_data = np.array(request.json["data"])

        # Perform prediction using the loaded model
        prediction = model["replay-attack"].predict_proba(input_data)

        # Return the prediction as a JSON response
        return jsonify({"prediction": prediction.tolist()}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/predict/print", methods=["POST"])
def predict_print():
    try:
        # Get the input data as a NumPy array from the request
        input_data = np.array(request.json["data"])

        # Perform prediction using the loaded model
        prediction = model["print-attack"].predict_proba(input_data)

        # Return the prediction as a JSON response
        return jsonify({"prediction": prediction.tolist()}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
