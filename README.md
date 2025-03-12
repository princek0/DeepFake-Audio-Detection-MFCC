# Deepfake Audio Detection Using MFCC Features

## Overview
This project implements a machine learning approach to detect AI-generated (deepfake) audio using MFCC (Mel-frequency cepstral coefficients) features. The implementation is based on the research paper by A. Hamza et al., using a Support Vector Machine (SVM) classifier to differentiate between genuine and deepfake audio samples.

This repository is a fork of the original project developed during the AIAmplify Hackathon. I have made several improvements to enhance the robustness and usability of the application.

## Research Foundation
**Paper**: "Deepfake Audio Detection via MFCC Features Using Machine Learning," in IEEE Access, vol. 10, pp. 134018-134028, 2022.

**Authors**: A. Hamza et al.

**DOI**: 10.1109/ACCESS.2022.3231480

**Abstract**: This research uses machine learning techniques to identify deepfake audio, focusing on Mel-frequency cepstral coefficients (MFCCs) to extract the most valuable audio information. The study employs the Fake-or-Real dataset created with text-to-speech models. Experimental results demonstrate that the support vector machine (SVM) outperformed other machine learning models in terms of accuracy on specific subsets of the dataset.

**URL**: [IEEE Access Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9996362)

## Original Project Contributors
- Noor Chauhan
- [Abhishek Khadgi](https://github.com/abhis-hek)
- Omkar Sapkal
- Himanshi Shinde
- Furqan Ali

## Improvements in This Fork
- Enhanced audio file processing to support both WAV and MP3 file formats
- Implemented robust error handling for audio file loading and feature extraction
- Integrated SoundFile library for better audio format compatibility
- Fixed "No module named 'aifc'" error that prevented processing certain files
- Created comprehensive documentation and setup instructions
- Added requirements.txt for easier dependency management
- Added standalone analysis script for easy integration with other applications

## Table of Contents
1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [How to Use](#how-to-use)
   - [Training the Model](#training-the-model)
   - [Using the Web Application](#using-the-web-application)
   - [Command Line Analysis](#command-line-analysis)
   - [Using the Analysis Module](#using-the-analysis-module)
4. [API Integration](#api-integration)
5. [Technical Details](#technical-details)
6. [License](#license)

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Setup Steps
1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/your-username/DeepFake-Audio-Detection-MFCC.git
   cd DeepFake-Audio-Detection-MFCC
   ```

2. Set up a virtual environment (recommended):
   ```bash
   # For Windows
   python -m venv venv
   venv\Scripts\activate

   # For Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure
- `main.py`: Script for training the SVM model and basic audio analysis
- `app.py`: Flask web application for user-friendly deepfake detection
- `analyze_audio.py`: Standalone module for analyzing audio files with the pre-trained model
- `real_audio/`: Directory for storing genuine audio samples
- `deepfake_audio/`: Directory for storing deepfake audio samples
- `templates/`: Contains HTML templates for the web interface
- `uploads/`: Temporary storage for uploaded audio files
- `test_audio.py`: Test script to verify audio processing capabilities

## How to Use

### Training the Model
1. Prepare your dataset:
   - Place genuine audio files in the `real_audio` directory
   - Place deepfake audio files in the `deepfake_audio` directory
   - The audio files should be in WAV or MP3 format

2. Run the training script:
   ```bash
   python main.py
   ```

   This will:
   - Extract MFCC features from all audio files
   - Split the data into training and testing sets
   - Train an SVM classifier
   - Save the model and scaler for future use
   - Prompt you to analyze an audio file (optional)

### Using the Web Application
1. Ensure you have trained the model first (which creates `svm_model.pkl` and `scaler.pkl`)

2. Start the Flask web server:
   ```bash
   python app.py
   ```

3. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

4. Through the web interface, you can:
   - Upload audio files (WAV or MP3 format)
   - Receive instant classification results
   - Process multiple files in succession

### Command Line Analysis
To analyze a specific audio file directly:

1. Run the analysis script with the path to the audio file:
   ```bash
   python analyze_audio.py path/to/your/audio.wav
   ```

2. The script will output:
   - Classification result (REAL or DEEPFAKE)
   - Confidence score (0-1)
   - Raw decision score
   - Interpretation of the results

### Using the Analysis Module
The `analyze_audio.py` script can also be imported as a module in your own Python applications:

```python
import analyze_audio

# Analyze an audio file
result = analyze_audio.analyze_file("path/to/audio.wav")

# Access the results
classification = result["classification"]  # "real" or "deepfake"
confidence = result["confidence"]  # float between 0 and 1
raw_score = result["raw_score"]  # decision function value

# Check for errors
if "error" in result:
    print(f"Error occurred: {result['error']}")
else:
    print(f"The audio is {classification} with {confidence:.2f} confidence")
```

The `analyze_file` function accepts the following parameters:
- `audio_path`: Path to the audio file (required)
- `model_path`: Path to the saved SVM model (default: "svm_model.pkl")
- `scaler_path`: Path to the saved scaler (default: "scaler.pkl")

## API Integration
The `analyze_audio.py` module is designed to be easily integrated with REST APIs or WebSocket services. The repository includes dependencies for FastAPI, which allows:

- Creating real-time audio analysis endpoints
- Building WebSocket connections for streaming audio analysis
- Developing scalable microservices for audio verification

A basic FastAPI integration example might look like:

```python
from fastapi import FastAPI, UploadFile, File
import analyze_audio
import tempfile
import os

app = FastAPI()

@app.post("/analyze/")
async def analyze_audio_file(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    # Analyze the audio
    result = analyze_audio.analyze_file(temp_path)
    
    # Clean up
    os.remove(temp_path)
    os.rmdir(temp_dir)
    
    return result
```

## Technical Details
- **Feature Extraction**: The system extracts 13 MFCC coefficients from audio files
- **Classification**: Uses a Support Vector Machine with a linear kernel
- **Audio Processing**: Utilizes SoundFile and Librosa libraries for robust audio handling
- **Performance**: The model typically achieves accuracy between 80-95% depending on the dataset quality
- **Error Handling**: The module includes comprehensive error handling and validation
- **Confidence Scoring**: Uses distance from decision boundary to estimate prediction confidence

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Acknowledgments
- Original authors of the research paper for their methodology
- Original project contributors from the AIAmplify Hackathon
- The open-source community for the libraries and tools that made this project possible
