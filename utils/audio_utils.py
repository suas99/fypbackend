import numpy as np
import librosa
import io

def extract_audio_features(file, sr=22050, n_mfcc=13, n_mels=128):
    """
    Extract audio features from the uploaded audio file
    """
    try:
        # Read audio file
        audio_bytes = file.read()
        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        # Extract Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Combine features
        features = np.concatenate([mfccs, mel_spec_db])
        
        # Normalize features
        features = (features - np.mean(features)) / np.std(features)
        
        return features
    except Exception as e:
        raise Exception(f"Error extracting audio features: {str(e)}") 