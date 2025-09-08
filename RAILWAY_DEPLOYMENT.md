# Railway Deployment Guide for Nahdlatul Ulama AI

## Prerequisites
1. **GitHub Account** - Code repository
2. **Railway Account** - Free tier deployment
3. **Groq API Key** - Free LLM access

## Deployment Steps

### 1. Prepare Repository
```bash
# Push your code to GitHub
git add .
git commit -m "Production backend ready for Railway"
git push origin main
```

### 2. Create Railway Project
1. Go to [Railway](https://railway.app)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your `nahdlatul-ulama-ai` repository

### 3. Configure Environment Variables
In Railway dashboard, add these environment variables:
```
GROQ_API_KEY = your_actual_groq_api_key_here
PRODUCTION = true
```

### 4. Deploy Configuration
Railway will automatically:
- Detect the `railway.json` configuration
- Set `PORT` environment variable
- Run the `startCommand` from railway.json
- Monitor health via `/health` endpoint

### 5. Verify Deployment
1. Wait for deployment to complete (3-5 minutes)
2. Check Railway logs for startup messages
3. Visit your Railway URL + `/health` to verify
4. Test the `/docs` endpoint for API documentation

## Expected Startup Sequence
```
🚀 Starting Nahdlatul Ulama AI on Railway...
✅ Running in PRODUCTION mode on Railway
🏭 Starting FULL PRODUCTION Nahdlatul Ulama AI...
📊 Processing ALL 15,673+ SQL files + methodology texts...
🎉 FULL PRODUCTION startup complete in XX.XX seconds!
🚀 Production backend ready on Railway!
```

## API Endpoints
- `GET /health` - Health check
- `GET /docs` - API documentation
- `POST /search` - Search Islamic texts
- `POST /ask` - Ask questions to AI
- `GET /methods` - Get methodology texts

## Troubleshooting

### Common Issues
1. **Deployment fails**: Check Railway logs for specific errors
2. **Health check fails**: Verify GROQ_API_KEY is set correctly
3. **Slow startup**: First deploy takes longer due to embeddings creation

### Performance Notes
- **First startup**: 60-120 seconds (processing 15,673+ files)
- **Subsequent restarts**: 30-60 seconds (cached embeddings)
- **Memory usage**: ~512MB during operation
- **Cold start**: Railway may pause inactive services

### Railway Limits (Free Tier)
- **Execution time**: 500 hours/month
- **Memory**: 512MB RAM
- **Bandwidth**: Unlimited
- **Sleep**: Services sleep after 10 minutes of inactivity

## Production URLs
After deployment, you'll get a Railway URL like:
```
https://your-app-name.railway.app
```

Update your frontend's API configuration to use this URL.

## Monitoring
- Railway dashboard shows real-time metrics
- Check `/health` endpoint for system status
- Monitor Railway logs for any issues

## Next Steps
1. Deploy to Railway
2. Update frontend API URL
3. Deploy frontend to Vercel
4. Test end-to-end functionality
