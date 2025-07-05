# Emotion Detection Backend

This is the backend service for the emotion detection system that processes handwriting, voice, and facial expressions to detect mental states (Depression, Anxiety, Stress, Normal).

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your trained models in the `models` directory:
- `emothaw_cnn_model.keras` (handwriting model)
- `cnn_bilstm_dass_voice_model.h5` (voice model)
- `mental_state_faces_model2.keras` (face analysis model)
- `label_encoder.pkl` (face model label encoder - optional)
- `scaler.pkl` (face model scaler - optional)

## Running the Server

Start the Flask development server:
```bash
python app.py
```

The server will run on `http://localhost:5000`

## API Endpoints

### Handwriting Analysis
- **Endpoint**: `/api/handwriting`
- **Method**: POST
- **Input**: Image file (multipart/form-data)
- **Response**: JSON with emotion prediction and confidence

### Voice Analysis
- **Endpoint**: `/api/voice`
- **Method**: POST
- **Input**: Audio file (multipart/form-data)
- **Response**: JSON with emotion prediction and confidence

### Face Analysis
- **Endpoint**: `/api/face`
- **Method**: POST
- **Input**: Image file (multipart/form-data)
- **Response**: JSON with mental state prediction, confidence, and extracted features

#### Face Analysis Response Format:
```json
{
  "mental_state": "Depression",
  "confidence": 0.85,
  "features": {
    "eye_aspect_ratio": 0.25,
    "brow_drop": 0.15,
    "lip_tightness": 0.30,
    "blink_count": 2,
    "emotion_encoded": 4
  }
}
```

## Testing

You can test the face analysis handler using the provided test script:
```bash
python test_face_handler.py
```

This will open your webcam and allow you to test real-time face analysis by pressing 'p' to make predictions.

## Error Handling

The API returns appropriate error messages and status codes:
- 400: Bad Request (missing file, no face detected)
- 500: Internal Server Error (processing error)

## Dependencies

The face analysis handler requires:
- OpenCV for image processing
- MediaPipe for facial landmark detection
- DeepFace for emotion analysis
- TensorFlow for model inference
- scikit-learn for data preprocessing 