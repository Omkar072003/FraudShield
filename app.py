from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np
import logging
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
import os

app = Flask(__name__)
# Enable CORS for all routes with support for OPTIONS preflight requests
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"]}})

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Directory to store uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables to store model, features, and latest file
model = None
feature_columns = None
latest_file = None

# Load the trained model and feature columns if they exist
def load_model_and_features():
    global model, feature_columns
    try:
        if os.path.exists("fraud_model_advanced.pkl") and os.path.exists("feature_columns.pkl"):
            with open("fraud_model_advanced.pkl", "rb") as f:
                model = pickle.load(f)
            with open("feature_columns.pkl", "rb") as f:
                feature_columns = pickle.load(f)
            logging.info("Model and feature columns loaded successfully")
        else:
            logging.info("No pre-trained model found. Will train a new one when data is available.")
    except Exception as e:
        logging.error(f"Failed to load model or features: {e}")
        raise

load_model_and_features()

def preprocess_transaction(data):
    """Preprocess the incoming transaction data to match training features."""
    try:
        logging.debug(f"Preprocessing data: {data}")
        df = pd.DataFrame([data])
        
        # Feature engineering
        df["hour_of_day"] = pd.to_datetime(df["time"], format="%H:%M", errors='coerce').dt.hour
        if df["hour_of_day"].isna().any():
            logging.warning("Invalid time format, filling with random hour")
            df["hour_of_day"] = df["hour_of_day"].fillna(np.random.randint(0, 24))
        
        df["day_of_week"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors='coerce').dt.dayofweek
        if df["day_of_week"].isna().any():
            logging.warning("Invalid date format, filling with random day")
            df["day_of_week"] = df["day_of_week"].fillna(np.random.randint(0, 7))
        
        df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
        df["amount_log"] = np.log1p(df["amount"])
        
        # Include additional columns from CSV if present
        for col in ["location", "type"]:
            if col in data:
                df[col] = data[col]
            else:
                df[col] = "unknown"  # Default value
        
        # Fill missing features with default values
        logging.debug(f"Expected features: {feature_columns}")
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0  # Default value for missing features
        
        processed_df = df[feature_columns]
        logging.debug(f"Processed data: {processed_df.to_dict(orient='records')}")
        return processed_df
    except Exception as e:
        logging.error(f"Preprocessing error: {e}")
        raise

@app.route("/predict", methods=["POST"])
def predict():
    """Predict if a transaction is fraudulent."""
    global model, feature_columns
    try:
        data = request.get_json()
        logging.debug(f"Received data: {data}")
        
        # Validate input
        required_fields = ["transaction_id", "amount", "date", "time"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400
        
        if not model or not feature_columns:
            return jsonify({"error": "Model not trained yet"}), 400
        
        # Preprocess the transaction
        processed_data = preprocess_transaction(data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        confidence = model.predict_proba(processed_data)[0].max() * 100
        logging.debug(f"Prediction: {prediction}, Confidence: {confidence}")
        
        # Determine risk level and color
        risk_level = "High" if confidence > 75 else "Medium" if confidence > 50 else "Low"
        color = "red" if confidence > 75 else "orange" if confidence > 50 else "green"
        
        # Risk factors and recommendations
        risk_factors = ["High amount"] if data["amount"] > 1000 else []
        recommendations = ["Review transaction"] if prediction == 1 else ["No action needed"]
        
        return jsonify({
            "prediction": "Fraud" if prediction == 1 else "Non-Fraudulent",
            "confidence": round(confidence, 2),
            "risk_level": risk_level,
            "color": color,
            "risk_factors": risk_factors,
            "recommendations": recommendations
        })
    except Exception as e:
        logging.error(f"Error in /predict: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/store_csv", methods=["POST", "OPTIONS"])
def store_csv():
    """Handle CSV file uploads."""
    global latest_file
    # Handle OPTIONS request explicitly
    if request.method == "OPTIONS":
        return "", 200
        
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "File must be a CSV"}), 400
        
        # Save the file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        latest_file = file_path
        logging.info(f"File saved successfully: {file_path}")
        
        return jsonify({"message": "File uploaded successfully"}), 200
    except Exception as e:
        logging.error(f"Error in /store_csv: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/get_csv_data", methods=["GET"])
def get_csv_data():
    """Return the uploaded CSV data as JSON."""
    global latest_file
    try:
        if not latest_file or not os.path.exists(latest_file):
            return jsonify({"error": "No file uploaded yet"}), 404
        
        df = pd.read_csv(latest_file)
        logging.info(f"Returning CSV data from: {latest_file}")
        return jsonify(df.to_dict(orient='records')), 200
    except Exception as e:
        logging.error(f"Error in /get_csv_data: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/train", methods=["POST"])
def train():
    """Train the fraud detection model using the uploaded CSV data."""
    global model, feature_columns, latest_file
    try:
        if not latest_file or not os.path.exists(latest_file):
            return jsonify({"error": "No file uploaded yet"}), 404
        
        # Load and preprocess the data
        df = pd.read_csv(latest_file)
        logging.debug(f"Loaded CSV data: {df.head().to_dict()}")
        
        # Feature engineering
        df["hour_of_day"] = pd.to_datetime(df["time"], format="%H:%M", errors='coerce').dt.hour
        df["hour_of_day"] = df["hour_of_day"].fillna(np.random.randint(0, 24))
        df["day_of_week"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors='coerce').dt.dayofweek
        df["day_of_week"] = df["day_of_week"].fillna(np.random.randint(0, 7))
        df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
        df["amount_log"] = np.log1p(df["amount"])
        
        # Validate target column
        if 'fraud' not in df.columns:
            return jsonify({"error": "No 'fraud' column found in CSV"}), 400
        
        X = df.drop(columns=['fraud'])
        y = df['fraud']
        feature_columns = X.columns.tolist()
        
        # Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Save the model and feature columns
        with open("fraud_model_advanced.pkl", "wb") as f:
            pickle.dump(model, f)
        with open("feature_columns.pkl", "wb") as f:
            pickle.dump(feature_columns, f)
        
        logging.info("Model trained and saved successfully")
        return jsonify({"message": "Model trained successfully"}), 200
    except Exception as e:
        logging.error(f"Error in /train: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/test", methods=["POST"])
def test():
    """Test a sample transaction against the trained model."""
    global model, feature_columns
    try:
        if not model or not feature_columns:
            return jsonify({"error": "Model not trained yet"}), 400
        
        data = request.get_json()
        logging.debug(f"Received test data: {data}")
        
        # Preprocess the transaction
        processed_data = preprocess_transaction(data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        confidence = model.predict_proba(processed_data)[0].max() * 100
        logging.debug(f"Test Prediction: {prediction}, Confidence: {confidence}, Features: {processed_data.to_dict()}")
        
        # Determine risk level and color
        risk_level = "High" if confidence > 75 else "Medium" if confidence > 50 else "Low"
        color = "red" if confidence > 75 else "orange" if confidence > 50 else "green"
        
        # Risk factors and recommendations
        risk_factors = ["High amount"] if data.get("amount", 0) > 1000 else []
        recommendations = ["Review transaction"] if prediction == 1 else ["No action needed"]
        
        return jsonify({
            "prediction": "Fraud" if prediction == 1 else "Non-Fraudulent",
            "confidence": round(confidence, 2),
            "risk_level": risk_level,
            "color": color,
            "risk_factors": risk_factors,
            "recommendations": recommendations
        })
    except Exception as e:
        logging.error(f"Error in /test: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)