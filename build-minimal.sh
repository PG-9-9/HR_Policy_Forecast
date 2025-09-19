#!/bin/bash
# Build minimal HR Assistant Docker image (Chat UI only)

echo "🚀 Building minimal HR Assistant Docker image..."

# Build the minimal image
docker build -t hr-assistant-minimal:latest .

# Show image size
echo "📏 Image size:"
docker images hr-assistant-minimal:latest

# Test the container
echo "🧪 Testing container..."
docker run --rm -d -p 8000:8000 --name hr-test hr-assistant-minimal:latest

# Wait for startup
sleep 10

# Test health endpoint
echo "🏥 Testing health endpoint..."
curl -f http://localhost:8000/health

# Stop test container
docker stop hr-test

echo "✅ Build completed successfully!"
echo "🌐 Run with: docker run -p 8000:8000 -e OPENAI_API_KEY=your-key hr-assistant-minimal:latest"