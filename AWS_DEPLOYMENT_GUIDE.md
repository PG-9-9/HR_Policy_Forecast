# AWS EC2 Deployment Guide for HR Policy Chatbot

## 🚀 Complete AWS EC2 Deployment Steps

### Prerequisites
- AWS Account with EC2 access
- AWS CLI installed locally
- Docker Hub account (optional, for image registry)

---

## 1. Launch EC2 Instance

### Instance Configuration:
```bash
# Recommended EC2 Instance
Instance Type: t3.large (2 vCPU, 8GB RAM)
# For production: t3.xlarge (4 vCPU, 16GB RAM)

OS: Amazon Linux 2 or Ubuntu 22.04 LTS
Storage: 20GB gp3 (minimum for 4GB Docker image)
```

### Security Group Rules:
```bash
# HTTP access
Type: HTTP, Protocol: TCP, Port: 80, Source: 0.0.0.0/0

# HTTPS access
Type: HTTPS, Protocol: TCP, Port: 443, Source: 0.0.0.0/0

# Custom application (if needed)
Type: Custom TCP, Protocol: TCP, Port: 8000, Source: 0.0.0.0/0

# SSH access
Type: SSH, Protocol: TCP, Port: 22, Source: Your IP
```

---

## 2. Connect to EC2 Instance

```bash
# SSH into your instance
ssh -i your-key.pem ec2-user@your-ec2-public-ip

# Or for Ubuntu
ssh -i your-key.pem ubuntu@your-ec2-public-ip
```

---

## 3. Install Dependencies on EC2

### For Amazon Linux 2:
```bash
# Update system
sudo yum update -y

# Install Docker
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Git
sudo yum install -y git

# Logout and login again for Docker group permissions
exit
```

### For Ubuntu:
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt update
sudo apt install -y docker-ce
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -a -G docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Git
sudo apt install -y git

# Logout and login again
exit
```

---

## 4. Deploy Application

### Method A: Clone Repository and Build
```bash
# SSH back into instance
ssh -i your-key.pem ec2-user@your-ec2-public-ip

# Clone your repository
git clone https://github.com/PG-9-9/HR_Policy_Forecast.git
cd HR_Policy_Forecast

# Build Docker image
docker build -t mm-hr-optimized -f Dockerfile.optimized .
```

### Method B: Use Pre-built Image (Recommended)
```bash
# If you push to Docker Hub first
docker pull your-dockerhub-username/mm-hr-optimized:latest
```

---

## 5. 🔐 SECURE API KEY MANAGEMENT

### ⚠️ NEVER put API keys in commands or URLs!

### Method 1: Environment File (Recommended)
```bash
# Create secure environment file
sudo mkdir -p /opt/chatbot
sudo chown ec2-user:ec2-user /opt/chatbot

# Create .env file with restricted permissions
cat > /opt/chatbot/.env << 'EOF'
OPENAI_API_KEY=sk-proj-your-actual-api-key-here
PORT=8000
ENVIRONMENT=production
EOF

# Secure the file
chmod 600 /opt/chatbot/.env
```

### Method 2: AWS Systems Manager Parameter Store (Production)
```bash
# Store API key in AWS Parameter Store
aws ssm put-parameter \
    --name "/chatbot/openai-api-key" \
    --value "sk-proj-your-actual-api-key-here" \
    --type "SecureString" \
    --description "OpenAI API Key for HR Chatbot"

# Create script to retrieve and run
cat > /opt/chatbot/start-secure.sh << 'EOF'
#!/bin/bash
export OPENAI_API_KEY=$(aws ssm get-parameter --name "/chatbot/openai-api-key" --with-decryption --query "Parameter.Value" --output text)
docker run -d -p 8000:8000 -e OPENAI_API_KEY="$OPENAI_API_KEY" --name hr-chatbot mm-hr-optimized
EOF

chmod +x /opt/chatbot/start-secure.sh
```

---

## 6. Run Application Securely

### Using Environment File:
```bash
# Run with environment file
docker run -d \
  -p 8000:8000 \
  --env-file /opt/chatbot/.env \
  --name hr-chatbot \
  --restart unless-stopped \
  mm-hr-optimized
```

### Using Docker Compose (Recommended):
```bash
# Create docker-compose.yml
cat > /opt/chatbot/docker-compose.yml << 'EOF'
version: '3.8'

services:
  hr-chatbot:
    image: mm-hr-optimized
    container_name: hr-chatbot
    ports:
      - "8000:8000"
    env_file:
      - .env
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
EOF

# Start with Docker Compose
cd /opt/chatbot
docker-compose up -d
```

---

## 7. Setup Reverse Proxy (Production)

### Install Nginx:
```bash
# Amazon Linux 2
sudo amazon-linux-extras install nginx1 -y

# Ubuntu
sudo apt install -y nginx
```

### Configure Nginx:
```bash
# Create nginx configuration
sudo tee /etc/nginx/conf.d/chatbot.conf << 'EOF'
server {
    listen 80;
    server_name your-domain.com;  # Replace with your domain
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support (if needed)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://localhost:8000/health;
        access_log off;
    }
}
EOF

# Test and restart nginx
sudo nginx -t
sudo systemctl start nginx
sudo systemctl enable nginx
```

---

## 8. Setup SSL with Let's Encrypt (Production)

```bash
# Install Certbot
sudo yum install -y python3-pip  # Amazon Linux
sudo pip3 install certbot certbot-nginx

# Or for Ubuntu
sudo apt install -y certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal (already set up by certbot)
sudo crontab -l | grep certbot
```

---

## 9. Monitoring and Logging

### Setup Log Monitoring:
```bash
# View Docker logs
docker logs hr-chatbot --follow

# View Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Setup Monitoring Script:
```bash
# Create monitoring script
cat > /opt/chatbot/monitor.sh << 'EOF'
#!/bin/bash
# Health check script

URL="http://localhost:8000/health"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $URL)

if [ $RESPONSE -eq 200 ]; then
    echo "$(date): Chatbot is healthy"
else
    echo "$(date): Chatbot is DOWN (HTTP $RESPONSE)"
    # Restart container
    docker restart hr-chatbot
fi
EOF

chmod +x /opt/chatbot/monitor.sh

# Add to crontab (check every 5 minutes)
(crontab -l ; echo "*/5 * * * * /opt/chatbot/monitor.sh >> /var/log/chatbot-monitor.log 2>&1") | crontab -
```

---

## 10. Firewall Configuration

```bash
# Amazon Linux 2 / RHEL
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload

# Ubuntu
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 22/tcp
sudo ufw --force enable
```

---

## 🎯 Final Deployment Commands Summary

```bash
# 1. SSH into EC2
ssh -i your-key.pem ec2-user@your-ec2-ip

# 2. Clone and build
git clone https://github.com/PG-9-9/HR_Policy_Forecast.git
cd HR_Policy_Forecast
docker build -t mm-hr-optimized -f Dockerfile.optimized .

# 3. Setup secure environment
sudo mkdir -p /opt/chatbot
echo "OPENAI_API_KEY=your-api-key-here" > /opt/chatbot/.env
chmod 600 /opt/chatbot/.env

# 4. Run securely
docker run -d -p 8000:8000 --env-file /opt/chatbot/.env --name hr-chatbot --restart unless-stopped mm-hr-optimized

# 5. Test
curl http://your-ec2-ip:8000/health
```

---

## 🔒 Security Best Practices

1. **Never expose API keys in:**
   - Command history
   - Environment variables visible to other users
   - Log files
   - Git repositories

2. **Always use:**
   - Environment files with restricted permissions
   - AWS Parameter Store for production
   - HTTPS in production
   - Proper firewall rules

3. **Monitor:**
   - Application health
   - Resource usage
   - Security logs

Your chatbot will be accessible at:
- **HTTP**: `http://your-ec2-public-ip:8000`
- **With domain**: `http://your-domain.com` (after nginx setup)
- **HTTPS**: `https://your-domain.com` (after SSL setup)