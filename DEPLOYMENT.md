# HR Assistant - AWS Deployment Guide

This guide covers deploying the HR Assistant application to AWS using various methods.

## 📋 Prerequisites

- AWS CLI configured with appropriate permissions
- Docker installed locally
- OpenAI API key

## 🚀 Quick Deployment Options

### Option 1: AWS App Runner (Easiest)
```bash
# 1. Build and push to ECR
./deploy-aws.sh

# 2. Create App Runner service via AWS Console
# - Connect to ECR repository
# - Set environment variable: OPENAI_API_KEY
# - Auto-scaling: 1-10 instances
```

### Option 2: ECS Fargate (Recommended)
```bash
# 1. Deploy infrastructure
aws cloudformation create-stack \
  --stack-name hr-assistant-infra \
  --template-body file://cloudformation-template.yml \
  --parameters ParameterKey=OpenAIApiKey,ParameterValue=your-openai-key \
  --capabilities CAPABILITY_IAM

# 2. Build and deploy application
./deploy-aws.sh
```

### Option 3: EKS (Advanced)
```bash
# 1. Create EKS cluster
eksctl create cluster --name hr-assistant --region us-east-1

# 2. Apply Kubernetes manifests
kubectl apply -f k8s/
```

## 🐳 Local Testing

```bash
# Test with Docker Compose
docker-compose up --build

# Access application
curl http://localhost:8000/health
```

## 📊 Production Considerations

### Scaling
- **App Runner**: Auto-scales 1-25 instances
- **ECS Fargate**: Configure auto-scaling policies
- **EKS**: Horizontal Pod Autoscaler

### Security
- ✅ Non-root container user
- ✅ Secrets Manager for API keys
- ✅ Security groups with minimal access
- ✅ Image vulnerability scanning

### Monitoring
- CloudWatch logs and metrics
- Application Load Balancer health checks
- Custom health endpoint `/health`

### Cost Optimization
- Use Fargate Spot for development
- Schedule scaling for business hours
- Set up CloudWatch alarms for cost monitoring

## 🌐 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `PORT` | Application port | No (default: 8000) |
| `PYTHONUNBUFFERED` | Python logging | No (default: 1) |

## 📈 Deployment Architecture

```
Internet → ALB → ECS Fargate Tasks → ECR
                    ↓
                CloudWatch ← Secrets Manager
```

## 🔧 Customization

### Resource Requirements
- **CPU**: 1 vCPU (can scale to 2-4 for high traffic)
- **Memory**: 2 GB (can scale to 4-8 GB)
- **Storage**: Ephemeral (data persisted externally)

### Networking
- **VPC**: Dedicated VPC with public subnets
- **Security**: ALB security group + ECS security group
- **DNS**: Route 53 for custom domain (optional)

## 🚨 Troubleshooting

### Common Issues
1. **Container fails to start**: Check CloudWatch logs
2. **Health check fails**: Verify `/health` endpoint
3. **High memory usage**: Monitor PyTorch model loading

### Useful Commands
```bash
# Check ECS service status
aws ecs describe-services --cluster hr-assistant-cluster --services hr-assistant-service

# View logs
aws logs tail /ecs/hr-assistant --follow

# Update service
aws ecs update-service --cluster hr-assistant-cluster --service hr-assistant-service --force-new-deployment
```

## 💰 Cost Estimation

**ECS Fargate (us-east-1)**:
- 2 tasks × 1 vCPU × 2 GB RAM
- ~$35-50/month for 24/7 operation
- +ALB: ~$16/month
- +Data transfer: ~$9/month per GB

**Total**: ~$60-75/month for production setup

## 🔐 Security Checklist

- [ ] OpenAI API key stored in Secrets Manager
- [ ] Container runs as non-root user
- [ ] Security groups with minimal permissions
- [ ] VPC with private subnets (if using RDS)
- [ ] ALB with HTTPS (certificate required)
- [ ] CloudTrail enabled for audit logging
- [ ] Regular security scans of container images

## 📞 Support

For deployment issues, check:
1. CloudWatch logs: `/ecs/hr-assistant`
2. ECS service events
3. ALB target group health
4. Security group configurations