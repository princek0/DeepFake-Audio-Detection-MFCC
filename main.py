"""
Deepfake Audio Detection - Main Training and Analysis Module

This script implements a machine learning approach to detect AI-generated (deepfake) audio
using MFCC (Mel-frequency cepstral coefficients) features and a Support Vector Machine (SVM) classifier.

The script performs the following main functions:
1. Extracts MFCC features from audio files in specified directories
2. Creates datasets for genuine and deepfake audio samples
3. Trains an SVM classifier on the extracted features
4. Saves the trained model and scaler for later use
5. Provides functionality to analyze new audio files

Based on research by A. Hamza et al., "Deepfake Audio Detection via MFCC Features Using Machine Learning"
"""

import os
import glob
import librosa
import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

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

def create_dataset(directory, label):
    """
    Create a dataset from audio files in a directory.
    
    Parameters:
    -----------
    directory : str
        Path to the directory containing audio files
    label : int
        Class label for the audio files (0 for genuine, 1 for deepfake)
        
    Returns:
    --------
    tuple (X, y)
        X: List of feature vectors
        y: List of corresponding labels
    """
    X, y = [], []  # Initialize empty lists for features and labels
    
    # Find all WAV files in the directory
    audio_files = glob.glob(os.path.join(directory, "*.wav"))
    
    # Process each audio file
    for audio_path in audio_files:
        # Extract MFCC features
        mfcc_features = extract_mfcc_features(audio_path)
        
        # If feature extraction was successful, add to dataset
        if mfcc_features is not None:
            X.append(mfcc_features)
            y.append(label)
        else:
            print(f"Skipping audio file {audio_path}")

    # Print dataset statistics
    print("Number of samples in", directory, ":", len(X))
    print("Filenames in", directory, ":", [os.path.basename(path) for path in audio_files])
    
    return X, y


def train_model(X, y):
    """
    Train an SVM classifier on the provided dataset.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix where each row is a sample and each column is a feature
    y : numpy.ndarray
        Target vector with class labels (0 for genuine, 1 for deepfake)
        
    Returns:
    --------
    None
        The trained model and scaler are saved to disk
    """
    # Check if we have at least two classes
    unique_classes = np.unique(y)
    print("Unique classes in y_train:", unique_classes)

    if len(unique_classes) < 2:
        raise ValueError("At least 2 classes are required to train the model")

    # Print dataset dimensions
    print("Size of X:", X.shape)
    print("Size of y:", y.shape)

    # Count samples in each class
    class_counts = np.bincount(y)
    
    # Check if we have enough samples for stratified splitting
    if np.min(class_counts) < 2:
        print("Combining both classes into one for training")
        X_train, y_train = X, y
        X_test, y_test = None, None
    else:
        # Split data into training and testing sets (80% train, 20% test)
        # stratify=y ensures that the class distribution is preserved in both sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Print dimensions of the split datasets
        print("Size of X_train:", X_train.shape)
        print("Size of X_test:", X_test.shape)
        print("Size of y_train:", y_train.shape)
        print("Size of y_test:", y_test.shape)

    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # If we have a test set, evaluate the model
    if X_test is not None:
        # Scale the test set using the same scaler
        X_test_scaled = scaler.transform(X_test)

        # Create and train the SVM classifier
        svm_classifier = SVC(kernel='linear', random_state=42)
        svm_classifier.fit(X_train_scaled, y_train)

        # Make predictions on the test set
        y_pred = svm_classifier.predict(X_test_scaled)

        # Calculate and print performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        confusion_mtx = confusion_matrix(y_test, y_pred)

        print("Accuracy:", accuracy)
        print("Confusion Matrix:")
        print(confusion_mtx)
    else:
        print("Insufficient samples for stratified splitting. Combining both classes into one for training.")
        print("Training on all available data.")

        # Create and train the SVM classifier on all data
        svm_classifier = SVC(kernel='linear', random_state=42)
        svm_classifier.fit(X_train_scaled, y_train)

    # Save the trained SVM model and scaler for later use
    model_filename = "svm_model.pkl"
    scaler_filename = "scaler.pkl"
    joblib.dump(svm_classifier, model_filename)
    joblib.dump(scaler, scaler_filename)
    print(f"Model saved as {model_filename}")
    print(f"Scaler saved as {scaler_filename}")

def analyze_audio(input_audio_path):
    """
    Analyze a single audio file to determine if it's genuine or deepfake.
    
    Parameters:
    -----------
    input_audio_path : str
        Path to the audio file to analyze
        
    Returns:
    --------
    None
        Prints the classification result
    """
    # Load the saved model and scaler
    model_filename = "svm_model.pkl"
    scaler_filename = "scaler.pkl"
    
    try:
        svm_classifier = joblib.load(model_filename)
        scaler = joblib.load(scaler_filename)
    except FileNotFoundError:
        print(f"Error: Model files ({model_filename} or {scaler_filename}) not found. Please train the model first.")
        return

    # Validate the input file
    if not os.path.exists(input_audio_path):
        print("Error: The specified file does not exist.")
        return
    elif not input_audio_path.lower().endswith(".wav"):
        print("Error: The specified file is not a .wav file.")
        return

    # Extract MFCC features from the input audio
    mfcc_features = extract_mfcc_features(input_audio_path)

    if mfcc_features is not None:
        # Scale the features using the saved scaler
        mfcc_features_scaled = scaler.transform(mfcc_features.reshape(1, -1))
        
        # Make a prediction
        prediction = svm_classifier.predict(mfcc_features_scaled)
        
        # Print the result
        if prediction[0] == 0:
            print("The input audio is classified as genuine.")
        else:
            print("The input audio is classified as deepfake.")
    else:
        print("Error: Unable to process the input audio.")

def main():
    """
    Main function to orchestrate the training process.
    """
    # Define directories containing genuine and deepfake audio samples
    genuine_dir = r"real_audio"
    deepfake_dir = r"deepfake_audio"

    # Create datasets for genuine and deepfake audio
    X_genuine, y_genuine = create_dataset(genuine_dir, label=0)  # 0 = genuine
    X_deepfake, y_deepfake = create_dataset(deepfake_dir, label=1)  # 1 = deepfake

    # Check if each class has at least two samples (required for stratified splitting)
    if len(X_genuine) < 2 or len(X_deepfake) < 2:
        print("Each class should have at least two samples for stratified splitting.")
        print("Combining both classes into one for training.")
    
    # Combine the genuine and deepfake datasets
    X = np.vstack((X_genuine, X_deepfake))  # Vertical stack of feature matrices
    y = np.hstack((y_genuine, y_deepfake))  # Horizontal stack of label vectors

    # Train the model on the combined dataset
    train_model(X, y)

if __name__ == "__main__":
    """
    Entry point of the script.
    """
    # Run the main training function
    main()

    # After training, prompt the user to analyze an audio file
    user_input_file = input("Enter the path of the .wav file to analyze: ")
    analyze_audio(user_input_file)
