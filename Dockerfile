# Base Docker image with FastAPI and Python 3.8
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all files from your DeepFake model repository
COPY . .

# Install Python dependencies
RUN pip install -r requirements.txt

# Run FastAPI WebSocket API
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
