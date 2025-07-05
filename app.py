from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from handlers.handwriting_handler import predict_from_image
from handlers.face_handler import (
    predict_from_image as predict_face_from_image,
    predict_from_frame,
    load_models,
)
from handlers.voice_handler import predict_from_audio

# ✅ Correct __name__ here
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Load face analysis models once at startup
try:
    load_models()
    print("✅ Face analysis models loaded successfully!")
except Exception as e:
    print(f"⚠ Warning: Could not load face models: {e}")

@app.route('/api/handwriting', methods=['POST'])
def handwriting_api():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image uploaded'}), 400
    result = predict_from_image(file)
    return jsonify(result)

@app.route('/api/face/static', methods=['POST'])
def face_static_api():
    """Static image analysis endpoint"""
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image uploaded'}), 400
    result = predict_face_from_image(file)
    return jsonify(result)

@app.route('/api/face/realtime', methods=['POST'])
def face_realtime_api():
    """Real-time frame analysis endpoint"""
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
        return jsonify({'error': str(e)}), 500

@app.route('/api/voice', methods=['POST'])
def voice_api():
    file = request.files.get('audio')
    if not file:
        return jsonify({'error': 'No audio uploaded'}), 400
    result = predict_from_audio(file)
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Voice analysis server is running'})

# ✅ Correct __name__ and __main__ check
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)