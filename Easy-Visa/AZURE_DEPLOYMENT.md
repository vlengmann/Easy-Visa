# Azure Deployment Guide for Easy Visa App

This guide will help you deploy your Dash application to Azure App Service.

## Prerequisites

1. Azure account (sign up at https://azure.microsoft.com/free/)
2. Azure CLI installed (https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
3. Git repository set up

## Deployment Steps

### Option 1: Deploy via Azure Portal (Easiest)

1. **Push your code to GitHub**
   - Make sure all files are committed to your GitHub repository
   - Include: app.py, requirements.txt, startup.txt, model/, data/, utils/

2. **Create Azure App Service**
   - Go to https://portal.azure.com
   - Click "Create a resource" > "Web App"
   - Fill in:
     - Name: choose a unique name (e.g., easy-visa-predictor)
     - Runtime stack: Python 3.11 or 3.12
     - Region: Choose closest to you
     - Pricing: Free F1 tier (for testing) or Basic B1 (recommended)

3. **Configure Deployment**
   - After creation, go to your App Service
   - Navigate to "Deployment Center"
   - Choose "GitHub" as source
   - Authorize and select your repository and branch
   - Save

4. **Configure Startup Command**
   - Go to "Configuration" > "General settings"
   - Startup Command: `gunicorn --bind=0.0.0.0 --timeout 600 app:server`
   - Save

5. **Wait for Deployment**
   - Go to "Deployment Center" to see build logs
   - Once complete, your app will be at: https://[your-app-name].azurewebsites.net

### Option 2: Deploy via Azure CLI

1. **Login to Azure**
   ```bash
   az login
   ```

2. **Create Resource Group**
   ```bash
   az group create --name easy-visa-rg --location eastus
   ```

3. **Create App Service Plan**
   ```bash
   az appservice plan create --name easy-visa-plan --resource-group easy-visa-rg --sku B1 --is-linux
   ```

4. **Create Web App**
   ```bash
   az webapp create --resource-group easy-visa-rg --plan easy-visa-plan --name [your-unique-app-name] --runtime "PYTHON:3.11"
   ```

5. **Configure Startup Command**
   ```bash
   az webapp config set --resource-group easy-visa-rg --name [your-app-name] --startup-file "gunicorn --bind=0.0.0.0 --timeout 600 app:server"
   ```

6. **Deploy from Git**
   ```bash
   az webapp deployment source config --name [your-app-name] --resource-group easy-visa-rg --repo-url https://github.com/[your-username]/[your-repo] --branch master --manual-integration
   ```

7. **Stream Logs** (to debug if needed)
   ```bash
   az webapp log tail --name [your-app-name] --resource-group easy-visa-rg
   ```

## Important Files

- **requirements.txt**: Lists all Python dependencies
- **startup.txt**: Contains the startup command for Azure
- **.deployment**: Tells Azure to build during deployment
- **app.py**: Main application file (must expose `server = app.server`)

## Troubleshooting

### App won't start
- Check logs in Azure Portal > App Service > Log stream
- Verify startup command is correct
- Ensure all dependencies are in requirements.txt

### Out of memory
- Upgrade to a higher tier (B1 or higher)
- Model files might be too large for Free tier

### Missing files
- Ensure model/, data/, and utils/ folders are committed to git
- Check .gitignore isn't excluding necessary files

## Post-Deployment

- Your app will be available at: `https://[your-app-name].azurewebsites.net`
- Set up custom domain if needed
- Enable HTTPS (automatic with Azure)
- Monitor performance in Azure Portal

## Cost Considerations

- **Free F1**: Limited resources, app sleeps when idle
- **Basic B1**: ~$13/month, always on, better performance (recommended)
- **Standard S1**: ~$70/month, auto-scaling, production-ready

## Security Notes

- Use Azure Key Vault for sensitive data
- Enable authentication if needed (Azure AD)
- Monitor costs in Azure Portal
