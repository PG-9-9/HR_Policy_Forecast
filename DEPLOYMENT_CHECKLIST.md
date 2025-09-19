# 🚀 AWS EC2 Deployment Checklist

## Pre-Deployment
- [ ] AWS account with EC2 access
- [ ] OpenAI API key ready
- [ ] Domain name (optional, for production)

## EC2 Setup
- [ ] Launch t3.large instance (Ubuntu 22.04 or Amazon Linux 2)
- [ ] Configure security groups (HTTP: 80, HTTPS: 443, SSH: 22)
- [ ] Download key pair for SSH access

## Server Setup
- [ ] SSH into EC2 instance
- [ ] Install Docker and Docker Compose
- [ ] Install Git and other dependencies
- [ ] Configure firewall rules

## Application Deployment
- [ ] Clone repository: `git clone https://github.com/PG-9-9/HR_Policy_Forecast.git`
- [ ] Build Docker image: `docker build -t mm-hr-optimized -f Dockerfile.optimized .`
- [ ] Create secure .env file with API key
- [ ] Run container: `docker run -d -p 8000:8000 --env-file .env --name hr-chatbot --restart unless-stopped mm-hr-optimized`
- [ ] Test health endpoint: `curl http://localhost:8000/health`

## Production Setup (Optional)
- [ ] Install and configure Nginx reverse proxy
- [ ] Setup SSL certificate with Let's Encrypt
- [ ] Configure domain DNS to point to EC2 IP
- [ ] Setup monitoring and log rotation

## Security Checklist
- [ ] API key stored securely (not in commands/logs)
- [ ] .env file has restricted permissions (chmod 600)
- [ ] Firewall properly configured
- [ ] SSH key properly secured
- [ ] Consider AWS Parameter Store for API keys

## Testing
- [ ] Health endpoint responds: `curl http://your-ec2-ip:8000/health`
- [ ] Web interface loads: `http://your-ec2-ip:8000`
- [ ] Chatbot responds to "hi"
- [ ] Chatbot answers immigration questions
- [ ] No model downloads at runtime

## Post-Deployment
- [ ] Monitor logs: `docker logs hr-chatbot --follow`
- [ ] Set up automated backups (if needed)
- [ ] Document access URLs and credentials
- [ ] Test auto-restart: `docker restart hr-chatbot`

---

## Quick Commands Reference

```bash
# Check container status
docker ps

# View logs
docker logs hr-chatbot --follow

# Restart container
docker restart hr-chatbot

# Update application
git pull
docker build -t mm-hr-optimized -f Dockerfile.optimized .
docker stop hr-chatbot
docker rm hr-chatbot
docker run -d -p 8000:8000 --env-file .env --name hr-chatbot --restart unless-stopped mm-hr-optimized

# Check system resources
free -h
df -h
top
```

## Troubleshooting

**Container won't start:**
- Check logs: `docker logs hr-chatbot`
- Verify .env file exists and has correct permissions
- Check if port 8000 is available: `netstat -tlnp | grep 8000`

**Can't access from browser:**
- Check security group allows HTTP on port 8000
- Verify container is running: `docker ps`
- Test locally first: `curl http://localhost:8000/health`

**Models downloading at runtime:**
- This indicates the Docker build didn't cache models properly
- Rebuild the image and ensure no errors in build process