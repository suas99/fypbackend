#!/usr/bin/env python3
"""
Test script for the face analysis handler
"""

import cv2
import numpy as np
from handlers.face_handler import predict_from_frame, load_models

def test_face_handler():
    """Test the face handler with a webcam frame"""
    try:
        print("Loading models...")
        load_models()
        print("Models loaded successfully!")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Press 'q' to quit, 'p' to predict")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Display the frame
            cv2.imshow("Test Face Handler", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                print("\nMaking prediction...")
                result = predict_from_frame(frame)
                
                if 'error' in result:
                    print(f"Error: {result['error']}")
                else:
                    print(f"Mental State: {result['mental_state']}")
                    print(f"Confidence: {result['confidence']:.3f}")
                    print(f"Features:")
                    for key, value in result['features'].items():
                        print(f"  {key}: {value}")
                    print()
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    test_face_handler() 