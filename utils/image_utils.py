import cv2
import numpy as np
from PIL import Image
import io

def preprocess_image(file):
    """
    Preprocess the uploaded image file for model input
    """
    try:
        # Read image file
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to numpy array
        image = np.array(image)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Resize to model input size (assuming 224x224)
        image = cv2.resize(image, (224, 224))
        
        # Normalize pixel values
        image = image.astype('float32') / 255.0
        
        # Add channel dimension if needed
        image = np.expand_dims(image, axis=-1)
        
        return image
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}") 