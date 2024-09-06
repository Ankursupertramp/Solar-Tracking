from flask import Flask, jsonify, request, send_from_directory
import joblib
import tensorflow as tf
import pandas as pd

# Load the trained ANN model and scaler
model = tf.keras.models.load_model('ANN_model.h5')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

def predict_tilt_angle(model, month, day, hour, temperature, humidity, ghi):
    try:
        # Create a DataFrame with the input values
        input_data = pd.DataFrame({
            'Month': [month],
            'Day': [day],
            'Hour': [hour],
            'Temperature': [temperature],
            'Relative Humidity': [humidity],
            'GHI': [ghi]
        })

        # Convert DataFrame to NumPy array before scaling
        input_array = input_data.to_numpy()
        input_scaled = scaler.transform(input_array)

        # Predict the tilt angle using the ANN model
        predicted_tilt_angle = model.predict(input_scaled)[0][0]

        # Adjust angle based on the hour
        if 7 <= hour < 13:
            predicted_tilt_angle = -predicted_tilt_angle

        return float(predicted_tilt_angle)
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

@app.route('/')
def home():
    return send_from_directory('', 'index.html')

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Retrieve query parameters with default values if not provided
        month = request.args.get('month', type=int)
        day = request.args.get('day', type=int)
        hour = request.args.get('hour', type=int)
        temperature = request.args.get('temperature', type=float)
        humidity = request.args.get('humidity', type=float)
        ghi = request.args.get('ghi', type=float)

        # Check for missing parameters
        if None in (month, day, hour, temperature, humidity, ghi):
            return jsonify({'error': 'Missing or invalid query parameters'}), 400

        tilt_angle = predict_tilt_angle(ann_model, month, day, hour, temperature, humidity, ghi)

        if tilt_angle is None:
            return jsonify({'error': 'Error in prediction'}), 500

        return jsonify({'angle': tilt_angle})

    except Exception as e:
        print(f"Error in /predict endpoint: {e}")
        return jsonify({'error': str(e)}), 500
