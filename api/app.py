from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "../models/best_random_forest.pkl"
model = joblib.load(MODEL_PATH)

# Define a route for health check
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"message": "Credit Scoring API is running"}), 200

# Define a route for predictions
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse JSON input
        input_data = request.get_json()

        # Extract input features
        recency = float(input_data["recency"])
        frequency = int(input_data["frequency"])
        monetary = float(input_data["monetary"])
        severity = float(input_data["severity"])

        # Format data for the model
        features = np.array([[recency, frequency, monetary, severity]])

        # Make predictions
        prediction = model.predict(features)
        probability = model.predict_proba(features)[:, 1]  # Probability of "Bad" risk

        # Return prediction and probability
        return jsonify({
            "prediction": "Bad Risk" if prediction[0] == 1 else "Good Risk",
            "probability": float(probability[0])
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
