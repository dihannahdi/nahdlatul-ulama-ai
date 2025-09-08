# Deployment Guide for Nahdlatul Ulama AI

Complete guide to deploy the Islamic jurisprudence AI system using free services.

## 🎯 Overview

- **Backend**: Railway (Free tier)
- **Frontend**: Vercel (Free tier)  
- **Database**: ChromaDB (Self-hosted)
- **LLM**: Groq/Gemini (Free APIs)
- **Code**: GitHub (Free)

## 🚀 Step-by-Step Deployment

### 1. Prepare GitHub Repository

```bash
# Initialize Git repository
git init
git add .
git commit -m "Initial commit: Nahdlatul Ulama AI"

# Create GitHub repository and push
git remote add origin https://github.com/yourusername/nahdlatul-ulama-ai.git
git push -u origin main
```

### 2. Deploy Backend to Railway

#### A. Sign up and Create Project
1. Go to [Railway](https://railway.app)
2. Sign up with GitHub
3. Click "New Project"
4. Select "Deploy from GitHub repo"
5. Choose your nahdlatul-ulama-ai repository

#### B. Configure Railway Service
1. Select the `backend` folder as root directory
2. Set environment variables:
   ```
   GROQ_API_KEY=your_groq_api_key
   GOOGLE_API_KEY=your_google_api_key
   PORT=8000
   ```

#### C. Railway will automatically:
- Detect Python project
- Install requirements.txt
- Start with: `uvicorn main:app --host 0.0.0.0 --port $PORT`

#### D. Get Backend URL
- Note your Railway backend URL: `https://your-app.railway.app`

### 3. Deploy Frontend to Vercel

#### A. Sign up and Import Project
1. Go to [Vercel](https://vercel.com)
2. Sign up with GitHub
3. Click "New Project"
4. Import your GitHub repository
5. Set root directory to `frontend`

#### B. Configure Environment Variables
```
NEXT_PUBLIC_API_URL=https://your-railway-backend.railway.app
```

#### C. Deploy Settings
- Framework Preset: Next.js
- Build Command: `npm run build`
- Output Directory: `.next` (default)

### 4. Process Data

#### A. Run Data Processing Locally
```bash
# Process Islamic texts into vector embeddings
cd data
python process_data.py --mode full
```

#### B. Upload Vector Database
- The processed ChromaDB will be in `./vector_db`
- Railway will persist this in the backend container

### 5. Get Free API Keys

#### A. Groq API (Recommended)
1. Go to [Groq Console](https://console.groq.com)
2. Sign up for free account
3. Create API key
4. Free tier: 15 requests/minute

#### B. Google Gemini API (Backup)
1. Go to [Google AI Studio](https://ai.google.dev)
2. Create API key
3. Free tier: 60 requests/minute

### 6. Test the Deployment

#### A. Backend Health Check
```bash
curl https://your-railway-backend.railway.app/health
```

#### B. Frontend Access
- Visit: `https://your-vercel-app.vercel.app`
- Test Islamic Q&A functionality
- Verify method selection works

## 🔧 Configuration Details

### Railway Environment Variables
```
GROQ_API_KEY=gsk_your_key_here
GOOGLE_API_KEY=your_google_key_here  
CHROMA_PERSIST_DIRECTORY=/app/chroma_db
PORT=8000
ALLOWED_ORIGINS=https://your-vercel-app.vercel.app
```

### Vercel Environment Variables
```
NEXT_PUBLIC_API_URL=https://your-railway-backend.railway.app
```

## 💰 Free Tier Limits

### Railway Free Tier
- **Execution Time**: 500 hours/month
- **Memory**: 512MB
- **Storage**: 1GB
- **Bandwidth**: Unlimited

### Vercel Free Tier  
- **Bandwidth**: 100GB/month
- **Build Time**: 6000 minutes/month
- **Serverless Functions**: 100 per day
- **Storage**: Unlimited static files

### LLM API Limits
- **Groq**: 15 requests/minute (free)
- **Gemini**: 60 requests/minute (free)

## 🚨 Troubleshooting

### Common Issues

#### 1. Railway Build Failures
```bash
# Check logs in Railway dashboard
# Ensure requirements.txt is correct
# Verify Python version compatibility
```

#### 2. Vercel Build Errors
```bash
# Check build logs
# Verify package.json dependencies
# Ensure environment variables are set
```

#### 3. API Connection Issues
```bash
# Check CORS settings in FastAPI
# Verify API URLs in environment variables
# Test API endpoints manually
```

#### 4. LLM API Errors
```bash
# Verify API keys are valid
# Check rate limits
# Test API keys independently
```

### Performance Optimization

#### A. Backend Optimization
- Use Redis for caching (Railway Redis addon)
- Optimize vector search parameters
- Implement request batching

#### B. Frontend Optimization
- Enable Vercel Edge Functions
- Implement client-side caching
- Optimize bundle size

## 📊 Monitoring

### Railway Monitoring
- Check service logs
- Monitor resource usage
- Set up health checks

### Vercel Analytics
- Enable Vercel Analytics
- Monitor Core Web Vitals
- Track user interactions

## 🔐 Security

### API Security
- Rate limiting implemented
- CORS configuration
- Input validation
- Environment variable security

### Islamic Content Security
- Authentic source verification
- Methodology compliance
- Cultural sensitivity checks

## 🎯 Scaling

### When to Scale
- Monthly active users > 1000
- API requests > 10,000/month
- Storage needs > 1GB

### Scaling Options
- Railway Pro plan ($5/month)
- Vercel Pro plan ($20/month)
- Dedicated vector database
- CDN for static assets

## 📈 Success Metrics

### Technical Metrics
- ✅ Backend uptime > 99%
- ✅ Response time < 3 seconds
- ✅ Build success rate > 95%

### Islamic Content Metrics
- ✅ Authentic methodology implementation
- ✅ Proper Arabic text rendering
- ✅ Cultural sensitivity compliance

## 🤝 Support

### Resources
- Railway Documentation
- Vercel Documentation  
- LangChain Documentation
- Islamic Jurisprudence References

### Community
- GitHub Issues
- Railway Discord
- Vercel Discord
- Islamic Tech Communities
