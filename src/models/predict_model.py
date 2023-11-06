from flask import Flask, request, jsonify
from joblib import load
import pandas as pd

# Load the trained RandomForest model
model = load('models/random_forest_model.joblib')

def single_prediction(diabetes_indicators, model):
    # Convert the dictionary of indicators into a DataFrame
    X = pd.DataFrame([diabetes_indicators])
    # Make prediction
    prediction = model.predict(X)
    return prediction

# Initialize Flask app
app = Flask('diabetes_prediction')

# Define the predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data with patient's diabetes indicators
        diabetes_indicators = request.get_json(force=True)

        # Make single prediction with JSON data
        prediction = single_prediction(diabetes_indicators, model)

        # Return prediction as JSON
        response = {'Diabetes Prediction': 'High Chance Of Diabetes' if prediction[0] == 1 else 'Low Chance Of Diabetes'}
        return jsonify(response)
    
    except Exception as e:
        # If an error occurs, return the error message
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
