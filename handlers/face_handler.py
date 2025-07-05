import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from deepface import DeepFace
import mediapipe as mp
import os

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

EYE_AR_THRESH = 0.25

class BlinkDetector:
    def __init__(self):
        self.blink_counter = 0
        self.eye_closed = False

    def update(self, ear):
        if ear < EYE_AR_THRESH and not self.eye_closed:
            self.blink_counter += 1
            self.eye_closed = True
        elif ear >= EYE_AR_THRESH and self.eye_closed:
            self.eye_closed = False
        return self.blink_counter

def map_emotion(emotion):
    """Map emotion string to numeric value"""
    mapping = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}
    return mapping.get(emotion.lower(), 6)

def extract_features(frame, blink_detector):
    """Extract facial features from frame"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    features = np.zeros(20)  # Match your trained feature count

    if result.multi_face_landmarks:
        lm = result.multi_face_landmarks[0].landmark
        try:
            # EAR (Eye Aspect Ratio)
            left_eye_idx = [362, 385, 387, 263, 373, 380]
            right_eye_idx = [33, 160, 158, 133, 153, 144]
            def ear(pts): return (
                np.linalg.norm(np.array(pts[1]) - np.array(pts[5])) +
                np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
            ) / (2.0 * np.linalg.norm(np.array(pts[0]) - np.array(pts[3])) + 1e-6)

            left_pts = [(lm[i].x, lm[i].y) for i in left_eye_idx]
            right_pts = [(lm[i].x, lm[i].y) for i in right_eye_idx]
            avg_ear = (ear(left_pts) + ear(right_pts)) / 2

            # Blink
            blink = blink_detector.update(avg_ear)

            # Brow Drop
            brow = np.mean([lm[i].y for i in [337, 336, 296, 334]])
            eye = np.mean([lm[i].y for i in [463, 414, 286]])
            brow_drop = (brow - eye) * 1000

            # Lip Tightness
            lip_w = lm[308].x - lm[78].x
            lip_h = lm[14].y - lm[13].y
            lip_tight = (lip_w / (lip_h + 1e-6)) * 100

            # Emotion
            try:
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion = analysis[0]['dominant_emotion']
            except:
                emotion = "neutral"
            emo_encoded = map_emotion(emotion)

            # Final 20 features: (You can replace extra zeros later with more AUs or tracking data)
            features = [avg_ear, brow_drop, lip_tight, blink, emo_encoded] + [0]*15

        except Exception as e:
            print("⛔ Feature extraction failed:", e)

    return np.array(features)

# Load model and encoders once
model = None
label_encoder = None
scaler = None

def load_models():
    """Load the trained model and encoders"""
    global model, label_encoder, scaler
    
    if model is None:
        model_path = "models/mental_state_faces_model2.keras"
        label_encoder_path = "models/label_encoder.pkl"
        scaler_path = "models/scaler.pkl"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = load_model(model_path)
        
        # Try to load encoder files, but provide fallback if they don't exist
        try:
            if os.path.exists(label_encoder_path):
                with open(label_encoder_path, "rb") as f:
                    label_encoder = pickle.load(f)
            else:
                print(f"Warning: Label encoder file not found: {label_encoder_path}")
                print("Using default label mapping. Please ensure your model was trained with compatible labels.")
                # Create a simple label encoder with common mental states
                from sklearn.preprocessing import LabelEncoder
                label_encoder = LabelEncoder()
                label_encoder.fit(['Depression', 'Anxiety', 'Stress', 'Normal'])  # Adjust based on your model's classes
        except Exception as e:
            print(f"Error loading label encoder: {e}")
            raise
        
        try:
            if os.path.exists(scaler_path):
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
            else:
                print(f"Warning: Scaler file not found: {scaler_path}")
                print("Using StandardScaler. Please ensure your model was trained with compatible scaling.")
                # Create a simple scaler
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                # Note: This scaler won't be fitted, so you may need to fit it with your training data
        except Exception as e:
            print(f"Error loading scaler: {e}")
            raise

def predict_from_image(file):
    """
    Process face image and return mental state prediction
    """
    try:
        # Load models if not already loaded
        load_models()
        
        # Read image
        if isinstance(file, str):
            # If file is a path
            frame = cv2.imread(file)
        else:
            # If file is a file object (e.g., from Flask upload)
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {'error': 'Could not read image'}, 400
        
        # Initialize blink detector
        blink_detector = BlinkDetector()
        
        # Extract features
        features = extract_features(frame, blink_detector)
        
        # Check if features were extracted successfully
        if np.all(features == 0):
            return {'error': 'No face detected or feature extraction failed'}, 400
        
        # Make prediction
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled, verbose=0)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        confidence = float(np.max(prediction))
        
        # Return prediction results
        return {
            'mental_state': predicted_label,
            'confidence': confidence,
            'features': {
                'eye_aspect_ratio': float(features[0]),
                'brow_drop': float(features[1]),
                'lip_tightness': float(features[2]),
                'blink_count': int(features[3]),
                'emotion_encoded': int(features[4])
            }
        }
        
    except Exception as e:
        return {'error': str(e)}, 500

def predict_from_frame(frame):
    """
    Process a single frame (numpy array) and return mental state prediction
    """
    try:
        # Load models if not already loaded
        load_models()
        
        if frame is None:
            return {'error': 'Invalid frame'}, 400
        
        # Initialize blink detector
        blink_detector = BlinkDetector()
        
        # Extract features
        features = extract_features(frame, blink_detector)
        
        # Check if features were extracted successfully
        if np.all(features == 0):
            return {'error': 'No face detected or feature extraction failed'}, 400
        
        # Make prediction
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled, verbose=0)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        confidence = float(np.max(prediction))
        
        # Return prediction results
        return {
            'mental_state': predicted_label,
            'confidence': confidence,
            'features': {
                'eye_aspect_ratio': float(features[0]),
                'brow_drop': float(features[1]),
                'lip_tightness': float(features[2]),
                'blink_count': int(features[3]),
                'emotion_encoded': int(features[4])
            }
        }
        
    except Exception as e:
        return {'error': str(e)}, 500 