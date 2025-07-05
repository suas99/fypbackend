import numpy as np
from tensorflow.keras.models import load_model
from utils.audio_utils import extract_audio_features

# Global model variable for lazy loading
model = None

def load_voice_model():
    """Load the voice model lazily"""
    global model
    if model is None:
        try:
            model = load_model('models/cnn_bilstm_dass_voice_model.h5')
            print("✅ Voice model loaded successfully!")
        except Exception as e:
            print(f"⚠️ Warning: Could not load voice model: {e}")
            print("Voice analysis will not be available.")
            model = None
    return model

def predict_from_audio(file):
    """
    Process audio file and return emotion prediction
    """
    try:
        # Load model if not already loaded
        voice_model = load_voice_model()
        if voice_model is None:
            return {'error': 'Voice model not available - compatibility issue'}, 503
        
        # Extract audio features
        features = extract_audio_features(file)
        
        # Make prediction
        prediction = voice_model.predict(np.expand_dims(features, axis=0))[0]
        
        # Define emotion classes
        classes = ['Depression', 'Anxiety', 'Stress']
        
        # Return prediction results
        return {
            'emotion': classes[np.argmax(prediction)],
            'confidence': float(np.max(prediction))
        }
    except Exception as e:
        return {'error': str(e)}, 500 