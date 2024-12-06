from flask import Flask, request, jsonify, render_template
import mysql.connector
import traceback
import joblib  # To load a saved model
import numpy as np
from ML_Models import predict_random_forest  # Import model prediction function if needed

model = joblib.load('random_forest_model.pkl')  # Load the trained Random Forest model

app = Flask(__name__)

# Connection to the MySQL Database
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host='127.0.0.1',
            user='root',
            password='cdss',
            database='CDSS Diabetes'
        )
        return connection
    except mysql.connector.Error as err:
        raise ConnectionError(f"Database connection failed: {err}")

# Add a new diabetic patient
@app.route('/add_patient', methods=['POST'])
def add_patient():
    try:
        data = request.get_json()  # Assuming data is sent in JSON format
        query = """INSERT INTO DatasetofDiabetes 
                   (ID, No_Pation, Gender, AGE, Urea, Cr, HbA1c, Chol, TG, HDL, LDL, VLDL, BMI, CLASS) 
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute(query, tuple(data.values()))
        connection.commit()
        cursor.close()
        connection.close()
        return jsonify({'message': "Patient added successfully"}), 201
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/')
def index():
    return render_template('index.html')  # This loads the front-end page

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = {key.lower(): value for key, value in request.get_json().items()}
        app.logger.debug("Received data: %s", data)  # Log the received data

        # Validate required keys
        required_keys = ['gender', 'age', 'urea', 'creatinine', 'hba1c', 'cholesterol', 
                         'tg', 'hdl', 'ldl', 'vldl', 'bmi']
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            return jsonify({"error": f"Missing keys in input: {missing_keys}"}), 400

        # Convert input data into a features array
        features = [
            data['gender'],
            data['age'],
            data['urea'],
            data['creatinine'],
            data['hba1c'],
            data['cholesterol'],
            data['tg'],
            data['hdl'],
            data['ldl'],
            data['vldl'],
            data['bmi'],
        ]

        # Predict using the Random Forest model
        prediction = model.predict([features])[0]

        # Map prediction to class label
        class_mapping = {0: 'Non-Diabetic', 1: 'Pre-Diabetic', 2: 'Diabetic'}
        result = class_mapping.get(prediction, 'Unknown')

        return jsonify({'prediction': result})

    except KeyError as e:
        return jsonify({"error": f"Missing key: {str(e)}"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Update patient information
@app.route('/update_patient', methods=['PUT'])
def update_patient():
    try:
        data = request.get_json()  # Assuming data is sent in JSON format
        query = """UPDATE DatasetofDiabetes SET No_Pation=%s, Gender=%s, AGE=%s, Urea=%s, Cr=%s, HbA1c=%s, Chol=%s, TG=%s, 
                   HDL=%s, LDL=%s, VLDL=%s, BMI=%s, CLASS=%s WHERE ID=%s"""
        
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute(query, tuple(data.values()))
        connection.commit()
        cursor.close()
        connection.close()
        return jsonify({'message': "Patient updated successfully"}), 200
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

# Remove a patient
@app.route('/remove_patient/<int:ID>', methods=['DELETE'])
def remove_patient(ID):
    try:
        query = "DELETE FROM DatasetofDiabetes WHERE ID = %s"
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute(query, (ID,))
        connection.commit()
        cursor.close()
        connection.close()
        return jsonify({'message': f"Patient with ID {ID} removed successfully"}), 200
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True)