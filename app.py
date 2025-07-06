from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import traceback

# Import handlers with try-except to prevent startup failures
try:
    from handlers.handwriting_handler import predict_from_image
except Exception as e:
    print(f"⚠️ Warning: Could not import handwriting handler: {e}")
    # Define fallback function
    def predict_from_image(file):
        return {'error': 'Handwriting analysis not available', 'details': str(e)}, 503

try:
    from handlers.face_handler import (
        predict_from_image as predict_face_from_image,
        predict_from_frame,
        load_models,
    )
except Exception as e:
    print(f"⚠️ Warning: Could not import face handler: {e}")
    # Define fallback functions
    def predict_face_from_image(file):
        return {'error': 'Face analysis not available', 'details': str(e)}, 503
    def predict_from_frame(frame):
        return {'error': 'Face analysis not available', 'details': str(e)}, 503
    def load_models():
        print("⚠️ Face models could not be loaded")

try:
    from handlers.voice_handler import predict_from_audio
except Exception as e:
    print(f"⚠️ Warning: Could not import voice handler: {e}")
    # Define fallback function
    def predict_from_audio(file):
        return {'error': 'Voice analysis not available', 'details': str(e)}, 503

# Create Flask app
app = Flask(__name__)

# Configure CORS to allow requests from any origin
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://127.0.0.1:5500",  # VS Code Live Server
            "http://localhost:5500",   # Alternative Live Server
            "http://127.0.0.1:5501",   # Live Server alternate port
            "http://127.0.0.1:5502",   # Live Server alternate port
            "http://localhost:3000",   # Common development server
            "http://localhost:8000",   # Python simple HTTP server
            "http://localhost:8080",   # Alternative HTTP server
            "null",                    # For file:// protocol
            "*"                        # Allow all origins as fallback
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Load face analysis models once at startup
try:
    load_models()
    print("✅ Face analysis models loaded successfully!")
except Exception as e:
    print(f"⚠ Warning: Could not load face models: {e}")

@app.route('/api/handwriting', methods=['POST', 'OPTIONS'])
def handwriting_api():
    if request.method == 'OPTIONS':
        return handle_preflight()
        
    try:
        file = request.files.get('image')
        if not file:
            return jsonify({'error': 'No image uploaded'}), 400
        result = predict_from_image(file)
        return jsonify(result)
    except Exception as e:
        print(f"Error in handwriting API: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/api/face/static', methods=['POST', 'OPTIONS'])
def face_static_api():
    """Static image analysis endpoint"""
    if request.method == 'OPTIONS':
        return handle_preflight()
        
    try:
        file = request.files.get('image')
        if not file:
            return jsonify({'error': 'No image uploaded'}), 400
        result = predict_face_from_image(file)
        return jsonify(result)
    except Exception as e:
        print(f"Error in face static API: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/api/face/realtime', methods=['POST', 'OPTIONS'])
def face_realtime_api():
    """Real-time frame analysis endpoint"""
    if request.method == 'OPTIONS':
        return handle_preflight()
        
    try:
        # Get base64 encoded frame data
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({'error': 'No frame data provided'}), 400
        
        # Remove data:image/jpeg;base64, or similar prefix
        image_data = data['frame'].split(',')[1] if ',' in data['frame'] else data['frame']
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is not None:
            result = predict_from_frame(frame)
            return jsonify(result)
        else:
            return jsonify({'error': 'Invalid frame data'}), 400
            
    except Exception as e:
        print(f"Error processing frame: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/api/voice', methods=['POST', 'OPTIONS'])
def voice_api():
    if request.method == 'OPTIONS':
        return handle_preflight()
        
    try:
        file = request.files.get('audio')
        if not file:
            return jsonify({'error': 'No audio uploaded'}), 400
        result = predict_from_audio(file)
        return jsonify(result)
    except Exception as e:
        print(f"Error in voice API: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    response = jsonify({'status': 'healthy', 'message': 'Mental health analysis server is running'})
    return response

def handle_preflight():
    """Handle CORS preflight requests"""
    response = jsonify({'status': 'ok'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# Run the app
if __name__ == '__main__':
    print("Starting Mental Health Analysis Backend Server...")
    print("API will be available at: http://localhost:5000")
    print("CORS is configured to allow requests from VS Code Live Server and other common origins")
    print("Press Ctrl+C to stop the server")
    app.run(debug=True, host='0.0.0.0', port=5000)