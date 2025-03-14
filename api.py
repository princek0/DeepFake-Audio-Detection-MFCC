"""
Deepfake Audio Detection - FastAPI WebSocket API

This module implements a WebSocket API for real-time deepfake audio detection using FastAPI.
It leverages the analyze_audio.py module to process audio data and return deepfake detection
results in real-time via WebSockets.

Features:
1. WebSocket endpoint for streaming audio detection
2. File upload endpoint for analyzing complete audio files
3. Integration with the pre-trained SVM model from analyze_audio.py
4. Error handling and logging
5. Concurrent connection handling with async/await

Usage:
    uvicorn api.api:app --reload
"""

import os
import base64
import json
import logging
import tempfile
import uuid
import sys
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add parent directory to path so we can import analyze_audio
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the analyze_audio module
import analyze_audio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("deepfake_audio_api")

# Initialize FastAPI app
app = FastAPI(
    title="Deepfake Audio Detection API",
    description="API for real-time detection of deepfake audio using MFCC features and SVM",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connection manager for handling multiple WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New connection. Total active: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Connection closed. Total active: {len(self.active_connections)}")

    async def send_message(self, websocket: WebSocket, message: Dict):
        await websocket.send_json(message)

# Initialize connection manager
manager = ConnectionManager()

# Paths to model files - use parent directory
MODEL_PATH = os.path.join(parent_dir, "svm_model.pkl")
SCALER_PATH = os.path.join(parent_dir, "scaler.pkl")

# Audio processing class
class AudioProcessor:
    @staticmethod
    async def process_audio_file(file_path: str) -> Dict[str, Any]:
        """
        Process an audio file using the analyze_audio module.
        
        Args:
            file_path: Path to the audio file to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Use the analyze_file function from analyze_audio.py with explicit model paths
            result = analyze_audio.analyze_file(file_path, model_path=MODEL_PATH, scaler_path=SCALER_PATH)
            
            # If an error occurred, log it
            if "error" in result:
                logger.error(f"Error analyzing audio: {result['error']}")
                
            return result
            
        except Exception as e:
            logger.exception(f"Error processing audio file: {str(e)}")
            return {"error": f"Failed to process audio: {str(e)}"}

    @staticmethod
    async def save_base64_audio(audio_data: str, file_ext: str = ".wav") -> str:
        """
        Save base64 encoded audio data to a temporary file.
        
        Args:
            audio_data: Base64 encoded audio data
            file_ext: File extension to use for the temporary file
            
        Returns:
            Path to the saved temporary file
        """
        try:
            # Remove header if present (e.g., "data:audio/wav;base64,")
            if "," in audio_data:
                audio_data = audio_data.split(",", 1)[1]
                
            # Decode base64 data
            decoded_data = base64.b64decode(audio_data)
            
            # Create a temporary file
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f"audio_{uuid.uuid4()}{file_ext}")
            
            # Write data to the file
            with open(temp_file, "wb") as f:
                f.write(decoded_data)
                
            logger.info(f"Audio data saved to temporary file: {temp_file}")
            return temp_file
            
        except Exception as e:
            logger.exception(f"Error saving base64 audio: {str(e)}")
            raise e

# API routes
@app.get("/")
async def root():
    """API root endpoint providing basic information about the service."""
    return {
        "name": "Deepfake Audio Detection API",
        "version": "1.0.0",
        "description": "API for real-time detection of deepfake audio",
        "endpoints": {
            "POST /analyze": "Upload an audio file for deepfake detection",
            "WebSocket /ws": "Stream audio data for real-time deepfake detection"
        }
    }

@app.post("/analyze")
async def analyze_audio_file(file: UploadFile = File(...)):
    """
    Analyze an uploaded audio file for deepfake detection.
    
    Args:
        file: The uploaded audio file (WAV or MP3)
        
    Returns:
        JSON response with analysis results
    """
    # Validate file type
    supported_formats = [".wav", ".mp3"]
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in supported_formats:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported audio format. Supported formats: {', '.join(supported_formats)}"
        )
    
    try:
        # Create a temporary file to save the uploaded audio
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, f"upload_{uuid.uuid4()}{file_ext}")
        
        # Save the uploaded file
        with open(temp_file, "wb") as f:
            f.write(await file.read())
            
        logger.info(f"Uploaded file saved to: {temp_file}")
        
        # Process the audio file
        result = await AudioProcessor.process_audio_file(temp_file)
        
        # Clean up the temporary file
        os.remove(temp_file)
        
        return result
        
    except Exception as e:
        logger.exception(f"Error processing uploaded file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio analysis.
    
    Expects messages in JSON format with the following structure:
    {
        "audio_data": "base64_encoded_audio_data",
        "file_type": ".wav" or ".mp3"
    }
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive JSON data from the WebSocket
            data = await websocket.receive_json()
            
            # Extract audio data and file type
            audio_data = data.get("audio_data")
            file_type = data.get("file_type", ".wav")
            
            if not audio_data:
                await manager.send_message(
                    websocket, 
                    {"error": "No audio data provided in the message"}
                )
                continue
                
            try:
                # Save the base64 audio data to a temporary file
                temp_file = await AudioProcessor.save_base64_audio(audio_data, file_type)
                
                # Process the audio file
                result = await AudioProcessor.process_audio_file(temp_file)
                
                # Clean up the temporary file
                os.remove(temp_file)
                
                # Send the results back to the client
                await manager.send_message(websocket, result)
                
            except Exception as e:
                logger.exception(f"Error processing WebSocket data: {str(e)}")
                await manager.send_message(
                    websocket, 
                    {"error": f"Failed to process audio: {str(e)}"}
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        
    except Exception as e:
        logger.exception(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)

# Run the application
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 