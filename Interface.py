from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained models (ensure you save your models using joblib after training them)
random_forest_model = joblib.load('random_forest_model.pkl')  # Only load Random Forest as per the latest change

# Load LabelEncoder and Scaler if needed
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Placeholder dataset statistics for mean values
data_stats = {
    "cholesterol": 4.86,
    "ldl": 2.61,
    "hdl": 1.20,
    "vldl": 1.85,
    "tg": 2.35,
    "urea": 5.12,
    "creatinine": 68.94,
    "hba1c": 8.28,
    "bmi": 29.59,
    "age": 53.53  # You can add more defaults for any missing feature
}

# Serve the CSS file from the templates folder
@app.route('/style.css')
def serve_style_csscss():
    return send_from_directory('templates', 'style.css')  # Serve from templates folder

# Serve the CSS file from the templates folder
@app.route('/index.css')
def serve_index_css():
    return send_from_directory('templates', 'index.css')  # Serve from templates folder

# Serve the images from the public folder
@app.route('/public/<filename>')
def serve_image(filename):
    return send_from_directory('public', filename)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = {key.lower(): value for key, value in request.get_json().items()}
        app.logger.debug("Received data: %s", data)  # Log the received data

        # Map the lowercase feature names to the uppercase ones expected by the model
        feature_mapping = {
            'gender': 'Gender',
            'age': 'AGE',
            'urea': 'Urea',
            'creatinine': 'Cr',
            'hba1c': 'HbA1c',
            'cholesterol': 'Chol',
            'tg': 'TG',
            'hdl': 'HDL',
            'ldl': 'LDL',
            'vldl': 'VLDL',
            'bmi': 'BMI'
        }

        # Check for missing fields and fill with averages if needed
        input_data = {}
        for key, value in data.items():
            if key == 'gender':
                # Map 'M' to 1 and 'F' to 0
                input_data[feature_mapping[key]] = 1 if value.upper() == 'M' else 0
            else:
                input_data[feature_mapping[key]] = float(value) if value not in [None, ""] else data_stats.get(feature_mapping[key].lower(), 0)

        # Ensure all required keys are in the request
        required_keys = list(feature_mapping.values())
        missing_keys = [key for key in required_keys if key not in input_data]
        if missing_keys:
            return jsonify({"error": f"Missing keys in input: {missing_keys}"}), 400

        # Convert input data into a DataFrame for preprocessing
        input_df = pd.DataFrame([input_data])

        # Preprocess the input data (scale numerical values)
        input_scaled = scaler.transform(input_df)

        # Make prediction using the Random Forest model
        random_forest_pred = random_forest_model.predict(input_scaled)

        # Map the prediction to a readable class
        class_mapping = {0: 'Non-Diabetic', 1: 'Pre-Diabetic', 2: 'Diabetic'}
        prediction = class_mapping[random_forest_pred[0]]

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
