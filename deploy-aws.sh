#!/bin/bash
# AWS ECR and ECS deployment script

set -e

# Configuration
AWS_REGION="us-east-1"
ECR_REPOSITORY="hr-assistant"
ECS_CLUSTER="hr-assistant-cluster"
ECS_SERVICE="hr-assistant-service"
IMAGE_TAG="latest"

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}"

echo "🚀 Deploying HR Assistant to AWS..."

# 1. Login to ECR
echo "📦 Logging into ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_URI}

# 2. Build and tag image
echo "🔨 Building Docker image..."
docker build -t ${ECR_REPOSITORY}:${IMAGE_TAG} .
docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} ${ECR_URI}:${IMAGE_TAG}

# 3. Push to ECR
echo "⬆️ Pushing to ECR..."
docker push ${ECR_URI}:${IMAGE_TAG}

# 4. Update ECS service
echo "🔄 Updating ECS service..."
aws ecs update-service \
    --cluster ${ECS_CLUSTER} \
    --service ${ECS_SERVICE} \
    --force-new-deployment \
    --region ${AWS_REGION}

# 5. Wait for deployment
echo "⏳ Waiting for deployment to complete..."
aws ecs wait services-stable \
    --cluster ${ECS_CLUSTER} \
    --services ${ECS_SERVICE} \
    --region ${AWS_REGION}

echo "✅ Deployment completed successfully!"
echo "🌐 Service URL: https://your-alb-domain.com"