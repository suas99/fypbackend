import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError as MSEMetric
from utils.image_utils import preprocess_image
import json
import os
import random

# Load model with custom objects and handle compatibility issues
custom_objects = {
    'mse': MeanSquaredError(),
    'mean_squared_error': MeanSquaredError(),
    'MSE': MSEMetric()
}

# Set to mock model by default to ensure functionality
model = "mock"
print("⚠️ Using mock handwriting model for testing purposes")

# Try loading the real model if available, but don't fail if it's not
def try_load_model():
    global model
    model_paths = [
        'models/emothaw_cnn_model.h5',
        'models/emothaw_cnn_model.keras',
        'models/handwriting_model.h5',
        'models/handwriting_model.keras'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                print(f"Attempting to load handwriting model from {path}...")
                if path.endswith('.h5'):
                    model = load_model(path, custom_objects=custom_objects, compile=False)
                else:
                    model = load_model(path, compile=False)
                print(f"✅ Handwriting model loaded successfully from {path}!")
                return True
            except Exception as e:
                print(f"⚠️ Failed to load {path}: {e}")
                continue
    
    print("⚠️ Could not load any handwriting model, using mock model")
    return False

# Try to load the model in the background, but don't block startup
try:
    try_load_model()
except Exception as e:
    print(f"⚠️ Error during model loading attempt: {e}")

def predict_from_image(file):
    """
    Process handwriting image and return emotion prediction
    """
    try:
        # If using mock model, return a test prediction
        if model == "mock":
            # Read and preprocess the image to ensure the pipeline works
            try:
                image = preprocess_image(file)
            except Exception as img_err:
                print(f"Warning: Image preprocessing failed: {img_err}")
            
            # Return a mock prediction for testing
            emotions = ['Depression', 'Anxiety', 'Stress']
            emotion = random.choice(emotions)
            confidence = random.uniform(0.7, 0.95)
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'note': 'Using mock model for testing'
            }
            
        # Read and preprocess the image
        image = preprocess_image(file)
        
        # Make prediction
        prediction = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
        
        # Define emotion classes
        classes = ['Depression', 'Anxiety', 'Stress']
        
        # Return prediction results
        return {
            'emotion': classes[np.argmax(prediction)],
            'confidence': float(np.max(prediction))
        }
    except Exception as e:
        print(f"Error in handwriting prediction: {e}")
        # Fallback to mock prediction on error
        emotions = ['Depression', 'Anxiety', 'Stress']
        return {
            'emotion': random.choice(emotions),
            'confidence': random.uniform(0.6, 0.8),
            'note': 'Fallback prediction due to error',
            'error_details': str(e)
        } 