# Deepfake Audio Detection WebSocket API

This API provides real-time deepfake audio detection capabilities via both WebSocket connections and HTTP endpoints. It's designed to be integrated into applications that need to verify the authenticity of audio content.

## Architecture

The API is built with FastAPI and leverages the `analyze_audio.py` module to process audio files and extract MFCC features. These features are then used with a pre-trained SVM model to classify audio as either genuine or deepfake.

Key components:
- FastAPI for API framework
- WebSockets for real-time communication
- Pre-trained SVM model for audio classification
- Comprehensive error handling and logging

## Prerequisites

- Python 3.7+
- Pre-trained model files (`svm_model.pkl` and `scaler.pkl`) in the root directory
- Dependencies listed in `requirements.txt`

## Installation

1. Ensure you have all the required dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure the model files are present in the root directory:
   - `svm_model.pkl`
   - `scaler.pkl`

## Running the API

Start the API server with:

```bash
uvicorn api.api:app --reload --host 0.0.0.0 --port 8000
```

This will start the server on all interfaces on port 8000, with auto-reload enabled for development. Note that the API is located in the `api` folder, hence the `api.api:app` module path.

## API Endpoints

### HTTP Endpoints

#### GET /

Returns information about the API and available endpoints.

**Response:**
```json
{
  "name": "Deepfake Audio Detection API",
  "version": "1.0.0",
  "description": "API for real-time detection of deepfake audio",
  "endpoints": {
    "POST /analyze": "Upload an audio file for deepfake detection",
    "WebSocket /ws": "Stream audio data for real-time deepfake detection"
  }
}
```

#### POST /analyze

Analyze an uploaded audio file and determine if it's genuine or deepfake.

**Request:**
- Content-Type: multipart/form-data
- Body: Form data with a file field named "file" containing the audio file (WAV or MP3)

**Response:**
```json
{
  "classification": "real" | "deepfake",
  "confidence": 0.95,
  "raw_score": -2.5
}
```

### WebSocket Endpoint

#### /ws

WebSocket endpoint for real-time audio analysis.

**Expected Message Format:**
```json
{
  "audio_data": "base64_encoded_audio_data",
  "file_type": ".wav" | ".mp3"
}
```

**Response Format:**
```json
{
  "classification": "real" | "deepfake",
  "confidence": 0.95,
  "raw_score": -2.5
}
```

## Error Handling

All endpoints include comprehensive error handling:

- HTTP endpoints return appropriate status codes with detailed error messages
- WebSocket connections send error messages in the same format as successful responses, with an additional "error" field

Example error response:
```json
{
  "error": "Failed to process audio: Invalid audio format"
}
```

## Client Examples

The `api` directory includes example clients:

1. `client_example.html` - Basic example for uploading files and testing WebSocket communication
2. `realtime_client.html` - Advanced client with audio recording and real-time visualization
3. `python_client.py` - Python client for programmatic API usage

To use these clients:
1. Start the API server
2. Open the HTML files in a web browser (for the HTML clients)
3. Run the Python client with `python api/python_client.py --file path/to/audio.wav`

## Integration Guide

### Integrating with Your Application

#### JavaScript/Web Example:
```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://your-server:8000/ws');

// Listen for messages
ws.onmessage = (event) => {
  const result = JSON.parse(event.data);
  console.log(`Classification: ${result.classification}, Confidence: ${result.confidence}`);
};

// Send audio data
function sendAudioForAnalysis(audioBlob) {
  const reader = new FileReader();
  reader.onload = () => {
    const base64data = reader.result;
    ws.send(JSON.stringify({
      audio_data: base64data,
      file_type: '.wav'
    }));
  };
  reader.readAsDataURL(audioBlob);
}
```

#### Python Client Example:
```python
import asyncio
import websockets
import json
import base64

async def analyze_audio(audio_file_path):
    # Read and encode audio file
    with open(audio_file_path, 'rb') as f:
        audio_data = base64.b64encode(f.read()).decode('utf-8')
    
    # Connect to WebSocket
    async with websockets.connect('ws://your-server:8000/ws') as websocket:
        # Send audio data
        await websocket.send(json.dumps({
            'audio_data': audio_data,
            'file_type': '.wav' if audio_file_path.endswith('.wav') else '.mp3'
        }))
        
        # Receive and parse result
        result = json.loads(await websocket.recv())
        return result

# Example usage
if __name__ == "__main__":
    result = asyncio.run(analyze_audio('path/to/audio.wav'))
    print(f"Classification: {result['classification']}")
    print(f"Confidence: {result['confidence']}")
```

## Troubleshooting

Common issues and solutions:

1. **Connection refused**
   - Ensure the API server is running
   - Verify the correct host and port

2. **Model not found errors**
   - Ensure `svm_model.pkl` and `scaler.pkl` exist in the root directory
   - Run `main.py` to train the model if these files don't exist

3. **Invalid audio format**
   - Only WAV and MP3 formats are supported
   - Ensure audio files are properly formatted

4. **Connection drops during large file uploads**
   - Consider chunking large audio files
   - For large files, use the HTTP endpoint instead of WebSocket

## Deployment Considerations

For production deployment:

1. **Security**
   - Use HTTPS/WSS with proper certificates
   - Set specific CORS origins instead of allowing all (`"*"`)
   - Implement authentication for sensitive deployments

2. **Scaling**
   - Consider using multiple workers (`--workers` with uvicorn)
   - Use a process manager like Gunicorn or Supervisor

3. **Monitoring**
   - Implement more robust logging
   - Consider adding health check endpoints
   - Monitor system resource usage 