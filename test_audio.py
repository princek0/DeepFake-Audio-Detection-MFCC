"""
Deepfake Audio Detection - Audio Processing Test Module

This script is used to test the audio processing capabilities of the system.
It attempts to load audio files using different libraries and extract MFCC features
to verify that the audio processing pipeline is working correctly.

The script:
1. Tests loading audio files with SoundFile and Librosa
2. Tests extracting MFCC features from the loaded audio
3. Runs tests on both genuine and deepfake audio samples

This is useful for debugging audio processing issues before running the main application.
"""

import os
import numpy as np
import soundfile as sf
import librosa

def test_extract_features(audio_path):
    """
    Test the audio processing pipeline on a single audio file.
    
    This function attempts to:
    1. Load the audio file using SoundFile
    2. If that fails, try loading with Librosa
    3. Extract MFCC features from the loaded audio
    
    Parameters:
    -----------
    audio_path : str
        Path to the audio file to test
        
    Returns:
    --------
    bool
        True if the entire pipeline succeeds, False otherwise
    """
    print(f"Testing file: {audio_path}")
    
    # Step 1: Test loading the audio file with SoundFile
    try:
        print("Trying with soundfile...")
        audio_data, sr = sf.read(audio_path)
        
        # Convert stereo to mono if needed
        if audio_data.ndim > 1:  # If stereo, convert to mono
            audio_data = np.mean(audio_data, axis=1)
            
        print(f"Success! Loaded with soundfile. Shape: {audio_data.shape}, Sample rate: {sr}")
    except Exception as e:
        # If SoundFile fails, log the error and try Librosa
        print(f"Soundfile failed: {e}")
        try:
            print("Trying with librosa...")
            audio_data, sr = librosa.load(audio_path, sr=None)
            print(f"Success! Loaded with librosa. Shape: {audio_data.shape}, Sample rate: {sr}")
        except Exception as e:
            # If both libraries fail, log the error and return failure
            print(f"Librosa failed: {e}")
            return False
    
    # Step 2: Test extracting MFCC features
    try:
        # Extract 13 MFCC coefficients
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        print(f"MFCC extraction successful. Shape: {mfccs.shape}")
        return True
    except Exception as e:
        # If MFCC extraction fails, log the error and return failure
        print(f"MFCC extraction failed: {e}")
        return False

# Main execution block
if __name__ == "__main__":
    # Get a list of files in the real_audio and deepfake_audio directories
    real_files = os.listdir("real_audio")
    deepfake_files = os.listdir("deepfake_audio")

    # Test a genuine audio file if available
    if real_files:
        real_file_path = os.path.join("real_audio", real_files[0])
        print("\n--- Testing real audio file ---")
        test_extract_features(real_file_path)

    # Test a deepfake audio file if available
    if deepfake_files:
        deepfake_file_path = os.path.join("deepfake_audio", deepfake_files[0])
        print("\n--- Testing deepfake audio file ---")
        test_extract_features(deepfake_file_path) 