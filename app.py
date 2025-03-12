"""
Deepfake Audio Detection - Web Application

This script implements a Flask web application that allows users to upload audio files
and determine whether they are genuine or AI-generated (deepfake) using a pre-trained
Support Vector Machine (SVM) model.

The application:
1. Provides a web interface for audio file upload
2. Processes uploaded audio files to extract MFCC features
3. Uses a pre-trained SVM model to classify the audio
4. Returns the classification result to the user

The model must be trained first using main.py before this application can be used.
"""

import os
from flask import Flask, request, render_template
import librosa
import numpy as np
import soundfile as sf
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# Initialize the Flask application
app = Flask(__name__)

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

def analyze_audio(input_audio_path):
    """
    Analyze an audio file to determine if it's genuine or deepfake.
    
    Parameters:
    -----------
    input_audio_path : str
        Path to the audio file to analyze
        
    Returns:
    --------
    str
        Classification result or error message
    """
    # Define paths to the saved model and scaler
    model_filename = "svm_model.pkl"
    scaler_filename = "scaler.pkl"

    # Validate the input file
    if not os.path.exists(input_audio_path):
        return "Error: The specified file does not exist."
    elif not (input_audio_path.lower().endswith(".wav") or input_audio_path.lower().endswith(".mp3")):
        return "Error: The specified file is not a supported audio format (WAV or MP3)."

    # Extract MFCC features from the input audio
    mfcc_features = extract_mfcc_features(input_audio_path)
    
    if mfcc_features is not None:
        try:
            # Load the scaler and scale the features
            scaler = joblib.load(scaler_filename)
            mfcc_features_scaled = scaler.transform(mfcc_features.reshape(1, -1))

            # Load the model and make a prediction
            svm_classifier = joblib.load(model_filename)
            prediction = svm_classifier.predict(mfcc_features_scaled)

            # Return the classification result
            if prediction[0] == 0:
                return "The input audio is classified as genuine."
            else:
                return "The input audio is classified as deepfake."
        except FileNotFoundError:
            return "Error: Model files not found. Please train the model first using main.py."
        except Exception as e:
            return f"Error during classification: {str(e)}"
    else:
        return "Error: Unable to process the input audio."

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Main route handler for the web application.
    
    Handles both GET requests (display the upload form) and
    POST requests (process uploaded audio files).
    
    Returns:
    --------
    str
        Rendered HTML template
    """
    # If the request is a POST, process the uploaded file
    if request.method == "POST":
        # Check if the post request has the file part
        if "audio_file" not in request.files:
            return render_template("index.html", message="No file part")
        
        # Get the file from the request
        audio_file = request.files["audio_file"]
        
        # If no file was selected, return an error
        if audio_file.filename == "":
            return render_template("index.html", message="No selected file")
        
        # If a valid file was uploaded, process it
        if audio_file and allowed_file(audio_file.filename):
            # Create the uploads directory if it doesn't exist
            if not os.path.exists("uploads"):
                os.makedirs("uploads")
            
            # Save the uploaded file
            audio_path = os.path.join("uploads", audio_file.filename)
            audio_file.save(audio_path)
            
            # Analyze the audio file
            result = analyze_audio(audio_path)
            
            # Clean up by removing the uploaded file
            os.remove(audio_path) 
            
            # Return the result page
            return render_template("result.html", result=result)
        
        # If the file format is not allowed, return an error
        return render_template("index.html", message="Invalid file format. Only .wav and .mp3 files allowed.")
    
    # If the request is a GET, display the upload form
    return render_template("index.html")

def allowed_file(filename):
    """
    Check if a filename has an allowed extension.
    
    Parameters:
    -----------
    filename : str
        Name of the file to check
        
    Returns:
    --------
    bool
        True if the file has an allowed extension, False otherwise
    """
    allowed_extensions = {'wav', 'mp3'}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions

if __name__ == "__main__":
    """
    Entry point of the script.
    """
    # Start the Flask development server
    app.run(debug=True)
