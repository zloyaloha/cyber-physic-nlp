#!/bin/bash
set -e

ollama serve &
OLLAMA_PID=$!

echo "Waiting for ollama to start..."
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 2
done
echo "Ollama is ready."

echo "Pulling qwen2.5:0.5b..."
ollama pull qwen2.5:0.5b
echo "Model ready."

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
