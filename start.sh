#!/bin/bash

# Simple startup script for cybersec-llm

echo "Starting Cybersecurity LLM Application..."

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "GOOGLE_API_KEY=" > .env
    echo "Created .env file."
fi

# Stop any existing containers
echo "Cleaning up any existing containers..."
docker-compose down

# Start Ollama first
echo "Starting Ollama service..."
docker-compose up -d ollama

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
max_attempts=30
attempt=1

while [ $attempt -le $max_attempts ]; do
    if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
        echo "✅ Ollama is ready!"
        break
    fi
    echo "Attempt $attempt/$max_attempts: Waiting for Ollama..."
    sleep 3
    attempt=$((attempt + 1))
done

if [ $attempt -gt $max_attempts ]; then
    echo "Ollama failed to start after $max_attempts attempts"
    exit 1
fi

# Pull required models
echo "Pulling required models..."
echo "Pulling llama3.2:latest (this may take a while)..."
docker exec cybersec-ollama ollama pull llama3.2:latest

echo "Pulling nomic-embed-text..."
docker exec cybersec-ollama ollama pull nomic-embed-text

# Verify models are available
echo "Verifying models..."
docker exec cybersec-ollama ollama list

# Start the application
echo "Starting the main application..."
docker-compose up -d cybersec-app

# Wait a moment and check status
sleep 5
echo ""
echo "Service Status:"
docker-compose ps

echo ""
echo "Setup complete!"
echo ""
echo "Application: http://localhost:8501"
echo "Ollama API: http://localhost:11434"
echo ""
echo "Useful commands:"
echo "  View app logs: docker-compose logs -f cybersec-app"
echo "  View all logs: docker-compose logs -f"
echo "  Stop all: docker-compose down"
echo "  Restart app: docker-compose restart cybersec-app" 