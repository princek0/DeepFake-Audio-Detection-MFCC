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

## Table of Contents
1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [How to Use](#how-to-use)
   - [Training the Model](#training-the-model)
   - [Using the Web Application](#using-the-web-application)
   - [Command Line Analysis](#command-line-analysis)
4. [Technical Details](#technical-details)
5. [License](#license)

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

1. Run the main script and provide the file path when prompted:
   ```bash
   python main.py
   # When prompted, enter the path to your audio file
   ```

## Technical Details
- **Feature Extraction**: The system extracts 13 MFCC coefficients from audio files
- **Classification**: Uses a Support Vector Machine with a linear kernel
- **Audio Processing**: Utilizes SoundFile and Librosa libraries for robust audio handling
- **Performance**: The model typically achieves accuracy between 80-95% depending on the dataset quality

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Acknowledgments
- Original authors of the research paper for their methodology
- Original project contributors from the AIAmplify Hackathon
- The open-source community for the libraries and tools that made this project possible
