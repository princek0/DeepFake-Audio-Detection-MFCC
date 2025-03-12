import os
import numpy as np
import soundfile as sf
import librosa

def test_extract_features(audio_path):
    print(f"Testing file: {audio_path}")
    try:
        # Using soundfile which has better support for various audio formats
        print("Trying with soundfile...")
        audio_data, sr = sf.read(audio_path)
        if audio_data.ndim > 1:  # If stereo, convert to mono
            audio_data = np.mean(audio_data, axis=1)
        print(f"Success! Loaded with soundfile. Shape: {audio_data.shape}, Sample rate: {sr}")
    except Exception as e:
        print(f"Soundfile failed: {e}")
        try:
            # Fall back to librosa if soundfile fails
            print("Trying with librosa...")
            audio_data, sr = librosa.load(audio_path, sr=None)
            print(f"Success! Loaded with librosa. Shape: {audio_data.shape}, Sample rate: {sr}")
        except Exception as e:
            print(f"Librosa failed: {e}")
            return False
    
    # Try to extract MFCC features
    try:
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        print(f"MFCC extraction successful. Shape: {mfccs.shape}")
        return True
    except Exception as e:
        print(f"MFCC extraction failed: {e}")
        return False

# Test with one real and one deepfake file
real_files = os.listdir("real_audio")
deepfake_files = os.listdir("deepfake_audio")

if real_files:
    real_file_path = os.path.join("real_audio", real_files[0])
    print("\n--- Testing real audio file ---")
    test_extract_features(real_file_path)

if deepfake_files:
    deepfake_file_path = os.path.join("deepfake_audio", deepfake_files[0])
    print("\n--- Testing deepfake audio file ---")
    test_extract_features(deepfake_file_path) 