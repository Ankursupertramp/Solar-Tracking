import os
import logging
from flask import Flask, jsonify, request, send_from_directory
import joblib
import tensorflow as tf
import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Load the trained ANN model and scaler
model = None
scaler = None

def load_model_and_scaler():
    global model, scaler
    try:
        model = tf.keras.models.load_model('ANN_model.h5')
        scaler = joblib.load('scaler.pkl')
        logging.info("Model and scaler loaded successfully")
    except Exception as e:
        logging.error(f"Error loading model or scaler: {e}")

load_model_and_scaler()

def predict_tilt_angle(model, month, day, hour, temperature, humidity, ghi):
    try:
        input_data = pd.DataFrame({
            'Month': [month],
            'Day': [day],
            'Hour': [hour],
            'Temperature': [temperature],
            'Relative Humidity': [humidity],
            'GHI': [ghi]
        })
        logging.debug(f"Input data: {input_data}")
        
        input_array = input_data.to_numpy()
        input_scaled = scaler.transform(input_array)
        logging.debug(f"Scaled input: {input_scaled}")
        
        predicted_tilt_angle = model.predict(input_scaled)[0][0]
        logging.debug(f"Raw predicted angle: {predicted_tilt_angle}")
        
        if 7 <= hour < 13:
            predicted_tilt_angle = -predicted_tilt_angle
        
        return float(predicted_tilt_angle)
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return None

@app.route('/')
def home():
    return send_from_directory('', 'index.html')

@app.route('/predict', methods=['GET'])
def predict():
    try:
        month = request.args.get('month', type=int)
        day = request.args.get('day', type=int)
        hour = request.args.get('hour', type=int)
        temperature = request.args.get('temperature', type=float)
        humidity = request.args.get('humidity', type=float)
        ghi = request.args.get('ghi', type=float)
        
        logging.debug(f"Received parameters: month={month}, day={day}, hour={hour}, temperature={temperature}, humidity={humidity}, ghi={ghi}")
        
        if None in (month, day, hour, temperature, humidity, ghi):
            return jsonify({'error': 'Missing or invalid query parameters'}), 400
        
        if model is None or scaler is None:
            load_model_and_scaler()
            if model is None or scaler is None:
                return jsonify({'error': 'Model or scaler not loaded'}), 500
        
        tilt_angle = predict_tilt_angle(model, month, day, hour, temperature, humidity, ghi)
        
        if tilt_angle is None:
            return jsonify({'error': 'Error in prediction'}), 500
        
        logging.debug(f"Predicted tilt angle: {tilt_angle}")
        return jsonify({'angle': tilt_angle})
    except Exception as e:
        logging.error(f"Error in /predict endpoint: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
