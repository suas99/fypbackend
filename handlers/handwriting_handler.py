import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError as MSEMetric
from utils.image_utils import preprocess_image

# Load model with custom objects
custom_objects = {
    'mse': MeanSquaredError(),
    'mean_squared_error': MeanSquaredError(),
    'MSE': MSEMetric()
}

# Load model once
model = load_model('models/emothaw_cnn_model.keras', custom_objects=custom_objects)

def predict_from_image(file):
    """
    Process handwriting image and return emotion prediction
    """
    try:
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
        return {'error': str(e)}, 500 