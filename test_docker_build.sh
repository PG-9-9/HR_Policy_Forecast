#!/bin/bash
# Quick test script for Linux Docker build

echo "🐳 Testing Linux Docker build..."

# Build the image
echo "📦 Building Docker image with Linux-compatible requirements..."
docker build -t mm-hr-chat-linux -f Dockerfile .

if [ $? -eq 0 ]; then
    echo "✅ Docker build successful!"
    
    echo "🧪 Running quick container test..."
    # Run container in background for testing
    docker run -d -p 8000:8000 --name mm-hr-test \
        -e OPENAI_API_KEY="test-key" \
        mm-hr-chat-linux
    
    sleep 10
    
    # Test health endpoint
    echo "🔍 Testing health endpoint..."
    if curl -f http://localhost:8000/health; then
        echo "✅ Health check passed!"
    else
        echo "❌ Health check failed"
    fi
    
    # Cleanup
    docker stop mm-hr-test
    docker rm mm-hr-test
    
else
    echo "❌ Docker build failed!"
    echo "Check the build logs above for Windows/Linux compatibility issues"
fi