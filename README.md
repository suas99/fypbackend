# Mental Health Analysis System

This system processes handwriting, voice, and facial expressions to detect mental states (Depression, Anxiety, Stress, Normal).

## Project Structure

- `app.py` - Flask backend for processing and analysis
- `index.html`, `styles.css`, `app.js` - Frontend web interface
- `start_backend.py` - Helper script to start the backend server

## Setup Instructions

### 1. Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Place Models in the `models` Directory

- `emothaw_cnn_model.h5` or `emothaw_cnn_model.keras` (handwriting model)
- `cnn_bilstm_dass_voice_model.h5` (voice model)
- `mental_state_faces_model2.keras` (face analysis model)
- `label_encoder.pkl` (face model label encoder - optional)
- `scaler.pkl` (face model scaler - optional)

## Running the Application

### 1. Start the Backend Server

```bash
# Option 1: Using the helper script
python start_backend.py

# Option 2: Directly running the Flask app
python app.py
```

The backend server will run on `http://localhost:5000`

### 2. Open the Frontend

Use VS Code Live Server extension or any other static file server:

1. In VS Code, right-click on `index.html`
2. Select "Open with Live Server"
3. The frontend will open in your browser (typically at `http://127.0.0.1:5500`)

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
- **Endpoint**: `/api/face/static` - For static images
- **Endpoint**: `/api/face/realtime` - For webcam frames
- **Method**: POST
- **Input**: Image file (multipart/form-data)
- **Response**: JSON with mental state prediction, confidence, and extracted features

## Testing

You can test the face analysis handler using the provided test script:
```bash
python test_face_handler.py
```

This will open your webcam and allow you to test real-time face analysis by pressing 'p' to make predictions.

## Troubleshooting

### CORS Issues
- The backend is configured to accept requests from common origins including VS Code Live Server
- If you encounter CORS issues, check that the backend server is running and accessible
- Verify that your frontend is making requests to `http://localhost:5000/api/*`

### Model Loading Issues
- If models fail to load, the system will use mock predictions for testing
- Check that your model files are in the correct format and located in the `models` directory
- Look for error messages in the console when starting the backend

### Connection Issues
- Make sure the backend server is running before opening the frontend
- Check that port 5000 is not being used by another application
- The frontend performs a health check to verify the backend is available 