#!/bin/bash

echo "--- 🚀 Launching AI Business Copilot Deployment ---"

# Step 1: Check for .env file
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create a .env file based on env.example and add your GROQ_API_KEY."
    exit 1
fi

# Step 2: Create persistent directories if they don't exist
mkdir -p uploads reports

# Step 3: Build and Start Containers
echo "🏗️ Building Docker containers..."
docker-compose down
docker-compose up --build -d

echo "✅ Deployment successful!"
echo "📍 Dashboard is now live at: http://localhost:8000"
echo "📊 Persistent data stored in ./uploads and ./reports"
