"""
Deepfake Audio Detection - Analysis Module

This script provides functionality to analyze audio files and determine if they are
genuine or AI-generated (deepfake) using a pre-trained Support Vector Machine (SVM) model.

The script:
1. Provides a function to analyze a single audio file
2. Returns classification result with confidence value
3. Can be used as a module in other scripts or as a command-line tool

Usage as a command-line tool:
    python analyze_audio.py <audio_file>

Usage as a module:
    import analyze_audio
    result = analyze_audio.analyze_file("path/to/audio.wav")
    print(result["classification"], result["confidence"])
"""

import os
import sys
import numpy as np
import soundfile as sf
import librosa
import joblib
from sklearn.preprocessing import StandardScaler

def extract_mfcc_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    """
    Extract MFCC features from an audio file.
    
    Parameters:
    -----------
    audio_path : str
        Path to the audio file
    n_mfcc : int, default=13
        Number of MFCC coefficients to extract
    n_fft : int, default=2048
        Length of the FFT window
    hop_length : int, default=512
        Number of samples between successive frames
        
    Returns:
    --------
    numpy.ndarray or None
        Mean of MFCC features if successful, None otherwise
    """
    try:
        # First attempt: Use soundfile which has better support for various audio formats
        audio_data, sr = sf.read(audio_path)
        if audio_data.ndim > 1:  # If stereo, convert to mono by averaging channels
            audio_data = np.mean(audio_data, axis=1)
    except Exception as e:
        try:
            # Second attempt: Fall back to librosa if soundfile fails
            audio_data, sr = librosa.load(audio_path, sr=None)
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return None

    # Extract MFCC features
    try:
        # Calculate MFCC features using librosa
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        # Return the mean of each coefficient across time to get a fixed-length feature vector
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error extracting MFCC features from {audio_path}: {e}")
        return None

def analyze_file(audio_path, model_path="svm_model.pkl", scaler_path="scaler.pkl"):
    """
    Analyze an audio file to determine if it's genuine or deepfake.
    
    Parameters:
    -----------
    audio_path : str
        Path to the audio file to analyze
    model_path : str, default="svm_model.pkl"
        Path to the saved SVM model
    scaler_path : str, default="scaler.pkl"
        Path to the saved StandardScaler
        
    Returns:
    --------
    dict
        Dictionary containing classification result and confidence:
        {
            "classification": "real" or "deepfake",
            "confidence": float value between 0 and 1,
            "error": Error message if any (only present if error occurred)
        }
    """
    result = {}
    
    # Validate input file
    if not os.path.exists(audio_path):
        result["error"] = f"Error: The specified file {audio_path} does not exist."
        return result
    
    # Check if file is a supported audio format (by extension)
    supported_extensions = ['.wav', '.mp3']
    file_ext = os.path.splitext(audio_path)[1].lower()
    if file_ext not in supported_extensions:
        result["error"] = f"Error: The file {audio_path} is not a supported audio format ({', '.join(supported_extensions)})."
        return result
    
    # Load model and scaler
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        result["error"] = f"Error: Model files ({model_path} or {scaler_path}) not found. Please train the model first."
        return result
    except Exception as e:
        result["error"] = f"Error loading model: {str(e)}"
        return result
    
    # Extract MFCC features
    mfcc_features = extract_mfcc_features(audio_path)
    if mfcc_features is None:
        result["error"] = "Error: Failed to extract MFCC features from the audio file."
        return result
    
    # Scale features
    mfcc_features_scaled = scaler.transform(mfcc_features.reshape(1, -1))
    
    # Make prediction
    prediction = model.predict(mfcc_features_scaled)[0]
    
    # Get distance from decision boundary as confidence measure
    # For binary classification, larger absolute values = more confidence
    decision_value = model.decision_function(mfcc_features_scaled)[0]
    
    # Convert to a confidence score (0 to 1)
    # We use a sigmoid-like function to map the distance to a probability
    confidence = 1.0 / (1.0 + np.exp(-np.abs(decision_value)))
    
    # Create result dictionary
    result["classification"] = "real" if prediction == 0 else "deepfake"
    result["confidence"] = float(confidence)
    result["raw_score"] = float(decision_value)
    
    return result

def print_result(result):
    """
    Print analysis result in a user-friendly format.
    
    Parameters:
    -----------
    result : dict
        Result dictionary from analyze_file function
    """
    if "error" in result:
        print(result["error"])
        return
    
    print("\n===== Audio Analysis Result =====")
    print(f"Classification: {result['classification'].upper()}")
    print(f"Confidence: {result['confidence']:.2f} ({int(result['confidence']*100)}%)")
    print(f"Raw decision score: {result['raw_score']:.4f}")
    print("=================================\n")
    
    # Interpretation message
    if result["confidence"] > 0.9:
        confidence_msg = "Very high confidence"
    elif result["confidence"] > 0.7:
        confidence_msg = "High confidence"
    elif result["confidence"] > 0.5:
        confidence_msg = "Moderate confidence"
    else:
        confidence_msg = "Low confidence"
    
    print(f"Interpretation: The audio is classified as {result['classification']} with {confidence_msg}.")
    print(f"A negative raw score indicates 'real', while a positive score indicates 'deepfake'.")
    if result["confidence"] < 0.6:
        print("Note: Low confidence results should be interpreted with caution.")

def main():
    """
    Main function to parse command line arguments and analyze audio file.
    """
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python analyze_audio.py <audio_file>")
        print("Example: python analyze_audio.py sample.wav")
        return
    
    # Get audio file path from command line
    audio_path = sys.argv[1]
    
    # Analyze audio file
    result = analyze_file(audio_path)
    
    # Print results
    print_result(result)

if __name__ == "__main__":
    """
    Entry point when script is run directly from command line.
    """
    main() 