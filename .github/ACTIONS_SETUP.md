# Simple GitHub Actions Setup for AWS EC2 Deployment

This repository includes a streamlined GitHub Actions workflow for automatic Docker build and deployment to AWS EC2.

## 🚀 What it Does

When you push code to the `main` or `master` branch:
1. **Builds** a Docker image using your `Dockerfile.optimized`
2. **Deploys** the image to your AWS EC2 instance
3. **Uses environment variables stored on EC2** (more secure)
4. **Verifies** the deployment with a health check

## 🔧 Setup Instructions

### **1. Add These Secrets to GitHub:**
Go to your repository → Settings → Secrets and variables → Actions → New repository secret:

```
AWS_PRIVATE_KEY = (content of your .pem key file)
AWS_HOST = your-ec2-ip-address (e.g., 3.15.123.45)  
AWS_USER = ec2-user (or ubuntu, depending on your AMI)
```

#### Required Secrets:
- `AWS_PRIVATE_KEY`: The content of your `.pem` key file (copy the entire file content)
- `AWS_HOST`: `3.15.123.45` or `my-app.example.com`
- `AWS_USER`: Usually `ec2-user`, `ubuntu`, or `admin` (depends on your AMI)

**Note**: We no longer need `OPENAI_API_KEY` in GitHub secrets since it will be stored directly on your EC2 instance.

### **2. Set Up Environment Variables on EC2:**

SSH into your EC2 instance and create a `.env` file:

```bash
# SSH into your EC2 instance
ssh -i your-key.pem ec2-user@your-ec2-ip

# Create environment file in your home directory
nano ~/.env

# Add your environment variables (replace with your actual API key):
OPENAI_API_KEY=sk-your-actual-openai-api-key-here

# Save and exit (Ctrl+X, then Y, then Enter)

# Make sure the file has correct permissions
chmod 600 ~/.env

# Verify the file was created correctly
cat ~/.env
```

**Security Benefits**:
- ✅ API key stored only on your EC2 instance
- ✅ No sensitive data in GitHub repository
- ✅ No API key transmitted through GitHub Actions
- ✅ Persists across deployments

### **3. Install Docker on EC2:**

```bash
# For Amazon Linux 2
sudo yum update -y
sudo yum install docker -y
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -a -G docker ec2-user

# For Ubuntu
sudo apt update
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -a -G docker ubuntu
```

**Important**: Log out and log back in after adding your user to the docker group.

## 🔄 How It Works

### Workflow Trigger
The workflow runs automatically when you:
- Push to `main` or `master` branch
- Manually trigger it from GitHub Actions tab

### Deployment Process
1. **Build**: Creates Docker image using `Dockerfile.optimized`
2. **Transfer**: Copies the image to your EC2 instance via SCP
3. **Deploy**: 
   - Stops any existing container
   - Loads the new image
   - Runs container with `--env-file ~/.env` (loads your environment variables)
4. **Verify**: Checks if the application is healthy from inside EC2

### Environment Variable Loading
- The container automatically loads all environment variables from `~/.env`
- Your OpenAI API key is available to the application
- No secrets are transmitted through GitHub Actions

## 🚨 Troubleshooting

### Common Issues

#### 1. Environment File Not Found
```
Error: file not found
```
**Solution**: 
- Make sure you created `~/.env` in your home directory
- Check file permissions: `ls -la ~/.env`
- Verify file contents: `cat ~/.env`

#### 2. OpenAI API Key Still Not Working
```
Error: The api_key client option must be set
```
**Solution**:
```bash
# Check if environment file exists and has correct format
cat ~/.env

# Make sure the API key starts with 'sk-'
# Example correct format:
# OPENAI_API_KEY=sk-1234567890abcdef...
```

#### 3. Docker Permission Denied
```
Got permission denied while trying to connect to the Docker daemon socket
```
**Solution**:
```bash
sudo usermod -a -G docker $USER
# Then log out and log back in
```

#### 4. Container Can't Read Environment File
```
Environment variables not loaded
```
**Solution**:
```bash
# Make sure the path is correct - use full path
ls -la /home/ec2-user/.env  # For ec2-user
ls -la /home/ubuntu/.env    # For ubuntu user
```

### Debugging Steps

1. **Check workflow logs** in GitHub Actions tab
2. **SSH into EC2** and check:
   ```bash
   # Check Docker status
   docker ps
   docker logs hr-policy-forecast
   
   # Check environment file
   cat ~/.env
   
   # Test environment loading
   docker run --rm --env-file ~/.env alpine env | grep OPENAI
   ```
3. **Test manually**:
   ```bash
   curl http://localhost:8000/health
   ```

## 📝 Customization

### Add More Environment Variables
Simply add them to your `~/.env` file:
```bash
# Edit the environment file
nano ~/.env

# Add more variables:
OPENAI_API_KEY=sk-your-key-here
DATABASE_URL=postgresql://...
DEBUG=false
API_TIMEOUT=30
```

### Change Application Port
If you want to use a different port:
1. Update the port in the workflow file (line with `-p 8000:8000`)
2. Update the health check URL
3. Ensure the new port is open in your EC2 security group

### Update Environment Variables
To update your API key or add new variables:
```bash
# SSH into EC2
ssh -i your-key.pem ec2-user@your-ec2-ip

# Edit environment file
nano ~/.env

# Save and exit, then redeploy or restart container
docker restart hr-policy-forecast
```

## 🔒 Security Benefits

✅ **API Key Security**: 
- Stored only on your EC2 instance
- Never transmitted through GitHub
- Not visible in workflow logs

✅ **Access Control**: 
- Only you have access via SSH
- File permissions protect the `.env` file
- Container inherits environment securely

✅ **Persistence**: 
- Environment variables persist across deployments
- No need to update GitHub secrets for API key changes
- Easy to manage and update

## 📊 Monitoring

After deployment, you can monitor your application:
- **Application**: `http://your-ec2-ip:8000`
- **Health check**: `http://your-ec2-ip:8000/health` (now shows OpenAI status)
- **Docker logs**: `ssh your-ec2 && docker logs hr-policy-forecast`
- **Environment check**: `ssh your-ec2 && docker exec hr-policy-forecast env | grep OPENAI`

Your deployment is now fully automated with secure environment variable management! 🎉