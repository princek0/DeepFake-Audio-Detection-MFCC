"""
Deepfake Audio Detection - Python WebSocket Client

This script demonstrates how to use the Deepfake Audio Detection WebSocket API from Python.
It includes examples for both WebSocket streaming and HTTP file uploads.

Usage:
    python api/python_client.py --file path/to/audio.wav [--method websocket|http]
"""

import argparse
import asyncio
import base64
import json
import os
import sys
from typing import Dict, Any, Optional

import aiohttp
import websockets


class DeepfakeAudioClient:
    """Client for the Deepfake Audio Detection API."""

    def __init__(self, base_url: str = "localhost:8000"):
        """
        Initialize the client with the API base URL.
        
        Args:
            base_url: Base URL of the API server (without protocol)
        """
        self.base_url = base_url
        self.http_url = f"http://{base_url}"
        self.ws_url = f"ws://{base_url}/ws"

    async def analyze_file_websocket(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze an audio file for deepfake detection using WebSocket.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary containing the analysis results
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine file type
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in ['.wav', '.mp3']:
            raise ValueError(f"Unsupported file format: {file_ext}. Only .wav and .mp3 are supported.")

        # Read and encode the file
        with open(file_path, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')

        # Connect to WebSocket
        try:
            async with websockets.connect(self.ws_url) as websocket:
                print(f"Connected to WebSocket at {self.ws_url}")
                
                # Prepare message
                message = {
                    'audio_data': audio_data,
                    'file_type': file_ext
                }
                
                # Send data
                print("Sending audio data...")
                await websocket.send(json.dumps(message))
                
                # Receive response
                print("Waiting for analysis results...")
                response = await websocket.recv()
                result = json.loads(response)
                
                return result
                
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"WebSocket connection closed unexpectedly: {e}")
            raise
        except Exception as e:
            print(f"Error during WebSocket communication: {e}")
            raise

    async def analyze_file_http(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze an audio file for deepfake detection using HTTP endpoint.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary containing the analysis results
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine file type
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in ['.wav', '.mp3']:
            raise ValueError(f"Unsupported file format: {file_ext}. Only .wav and .mp3 are supported.")

        try:
            # Prepare multipart form data
            url = f"{self.http_url}/analyze"
            
            async with aiohttp.ClientSession() as session:
                print(f"Uploading file to {url}")
                
                # Create form data with file
                data = aiohttp.FormData()
                data.add_field('file',
                               open(file_path, 'rb'),
                               filename=os.path.basename(file_path),
                               content_type='audio/wav' if file_ext == '.wav' else 'audio/mp3')
                
                # Send request
                async with session.post(url, data=data) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"HTTP error {response.status}: {error_text}")
                    
                    # Parse response
                    result = await response.json()
                    return result
                    
        except aiohttp.ClientError as e:
            print(f"HTTP request error: {e}")
            raise
        except Exception as e:
            print(f"Error during HTTP communication: {e}")
            raise

    def print_result(self, result: Dict[str, Any]) -> None:
        """
        Print the analysis results in a user-friendly format.
        
        Args:
            result: Dictionary containing the analysis results
        """
        if "error" in result:
            print(f"Error: {result['error']}")
            return
            
        classification = result["classification"].upper()
        confidence = result["confidence"] * 100
        raw_score = result["raw_score"]
        
        print("\n===== Audio Analysis Result =====")
        print(f"Classification: {classification}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Raw decision score: {raw_score:.4f}")
        print("=================================\n")
        
        # Interpretation
        if confidence > 90:
            confidence_msg = "very high confidence"
        elif confidence > 70:
            confidence_msg = "high confidence"
        elif confidence > 50:
            confidence_msg = "moderate confidence"
        else:
            confidence_msg = "low confidence"
            
        print(f"Interpretation: The audio is classified as {classification} with {confidence_msg}.")
        print(f"A negative raw score indicates 'real', while a positive score indicates 'deepfake'.")
        if confidence < 60:
            print("Note: Low confidence results should be interpreted with caution.")


async def main():
    """Main function to parse arguments and run the client."""
    parser = argparse.ArgumentParser(description="Deepfake Audio Detection Client")
    parser.add_argument("--file", required=True, help="Path to the audio file to analyze")
    parser.add_argument("--method", choices=["websocket", "http"], default="websocket",
                        help="Method to use for analysis (default: websocket)")
    parser.add_argument("--server", default="localhost:8000", help="API server address (default: localhost:8000)")
    
    args = parser.parse_args()
    
    client = DeepfakeAudioClient(args.server)
    
    try:
        if args.method == "websocket":
            result = await client.analyze_file_websocket(args.file)
        else:
            result = await client.analyze_file_http(args.file)
            
        client.print_result(result)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
        

if __name__ == "__main__":
    asyncio.run(main()) 