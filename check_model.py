#!/usr/bin/env python3
"""
Check mental state faces model details
"""

import tensorflow as tf
import numpy as np

def check_model():
    try:
        # Load the model
        model = tf.keras.models.load_model('models/mental_state_faces_model2.keras')
        
        print("✅ Model loaded successfully!")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
        print(f"Number of classes: {model.output_shape[-1]}")
        
        # Test with dummy data
        input_shape = model.input_shape[1:]  # Remove batch dimension
        dummy_input = np.random.random((1,) + input_shape)
        
        print(f"\nTesting with dummy input shape: {dummy_input.shape}")
        prediction = model.predict(dummy_input, verbose=0)
        print(f"Prediction shape: {prediction.shape}")
        print(f"Prediction: {prediction[0]}")
        print(f"Predicted class: {np.argmax(prediction[0])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    check_model() 