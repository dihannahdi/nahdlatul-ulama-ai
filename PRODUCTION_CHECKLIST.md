# Production Checklist for Nahdlatul Ulama AI

## ✅ Pre-Deployment Checklist

### 1. Code Quality & Testing
- [ ] Run backend tests: `python test_backend.py`
- [ ] Format code: `black backend/` and `isort backend/`
- [ ] Lint frontend: `cd frontend && npm run lint`
- [ ] Type check: `cd frontend && npm run type-check`
- [ ] Build frontend: `cd frontend && npm run build`

### 2. Environment Configuration
- [ ] Backend `.env` with all required variables
- [ ] Frontend `.env.local` with API URL
- [ ] Railway environment variables set
- [ ] Vercel environment variables set
- [ ] GitHub secrets configured for CI/CD

### 3. API Keys & Services
- [ ] Groq API key obtained and tested
- [ ] Google Gemini API key obtained and tested
- [ ] Keys added to production environment
- [ ] Rate limits understood and documented

### 4. Data Processing
- [ ] Islamic texts processed into vector embeddings
- [ ] ChromaDB populated with reference texts
- [ ] SQL chunks processed and embedded
- [ ] Vector search tested locally

### 5. Deployment Infrastructure
- [ ] Railway project created and configured
- [ ] Vercel project created and configured
- [ ] GitHub repository set up with CI/CD
- [ ] Domain names configured (if custom domains)

## 🚀 Deployment Steps

### Step 1: Backend Deployment (Railway)
1. **Create Railway Project**
   ```bash
   # Push code to GitHub first
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Configure Railway**
   - Connect GitHub repository
   - Set root directory to `backend/`
   - Add environment variables:
     ```
     GROQ_API_KEY=your_key_here
     GOOGLE_API_KEY=your_key_here
     PORT=8000
     CHROMA_PERSIST_DIRECTORY=/app/chroma_db
     ```

3. **Deploy and Test**
   - Railway will auto-deploy on push
   - Test health endpoint: `https://your-app.railway.app/health`
   - Note the Railway URL for frontend configuration

### Step 2: Frontend Deployment (Vercel)
1. **Create Vercel Project**
   - Import from GitHub
   - Set root directory to `frontend/`
   - Framework preset: Next.js

2. **Configure Environment**
   ```bash
   NEXT_PUBLIC_API_URL=https://your-railway-app.railway.app
   ```

3. **Deploy and Test**
   - Vercel will auto-deploy
   - Test the chat interface
   - Verify API connectivity

### Step 3: CI/CD Setup
1. **GitHub Secrets**
   ```
   RAILWAY_TOKEN=your_railway_token
   VERCEL_TOKEN=your_vercel_token
   VERCEL_ORG_ID=your_org_id
   VERCEL_PROJECT_ID=your_project_id
   ```

2. **Test Workflow**
   - Push a small change
   - Verify GitHub Actions run
   - Check both deployments update

## 🔍 Post-Deployment Testing

### Backend Tests
```bash
# Test all endpoints
curl https://your-railway-app.railway.app/health
curl https://your-railway-app.railway.app/methods
curl -X POST https://your-railway-app.railway.app/search \
  -H "Content-Type: application/json" \
  -d '{"query": "prayer", "limit": 3}'
```

### Frontend Tests
- [ ] Load the Vercel URL successfully
- [ ] Chat interface renders correctly
- [ ] Method selector works
- [ ] Can send messages and receive responses
- [ ] Islamic content displays properly
- [ ] Responsive design on mobile

### Integration Tests
- [ ] Frontend can communicate with backend
- [ ] CORS is configured correctly
- [ ] Error handling works properly
- [ ] Loading states display correctly

## 📊 Performance Monitoring

### Metrics to Track
- **Response Times**: API < 5s, UI interactions < 1s
- **Error Rates**: < 5% error rate
- **Uptime**: > 99% availability
- **Rate Limits**: Stay within free tier limits

### Monitoring Tools
- Railway: Built-in metrics and logs
- Vercel: Analytics and Web Vitals
- Browser: DevTools performance monitoring
- Manual: Regular functionality testing

## 🛡️ Security & Compliance

### Technical Security
- [ ] Environment variables secure
- [ ] No API keys in frontend code
- [ ] CORS properly configured
- [ ] Input validation enabled
- [ ] Rate limiting active

### Islamic Content Compliance
- [ ] Source attribution maintained
- [ ] Methodology authenticity preserved
- [ ] Cultural sensitivity reviewed
- [ ] Arabic text rendering correct

## 💡 Optimization Recommendations

### Performance
1. **Backend Optimizations**
   - Implement Redis caching on Railway
   - Optimize vector search parameters
   - Add request compression

2. **Frontend Optimizations**
   - Enable Vercel Edge Functions
   - Implement proper loading states
   - Optimize bundle size

### Cost Management
1. **Stay Within Free Tiers**
   - Monitor Railway hours (500/month)
   - Track Vercel bandwidth (100GB/month)
   - Respect API rate limits

2. **Scaling Indicators**
   - Users > 1000/month → Consider paid plans
   - Requests > 10,000/month → Optimize or upgrade
   - Storage > 1GB → Consider dedicated database

## 🆘 Troubleshooting Guide

### Common Issues

1. **Backend Won't Start**
   - Check Railway logs
   - Verify environment variables
   - Test locally first

2. **Frontend Can't Connect**
   - Check CORS configuration
   - Verify API URL environment variable
   - Test API endpoints manually

3. **LLM API Errors**
   - Verify API keys are correct
   - Check rate limits
   - Test keys independently

4. **Vector Search Issues**
   - Ensure data is processed
   - Check ChromaDB persistence
   - Verify embedding model

### Emergency Contacts
- **Technical Issues**: Check GitHub issues
- **Islamic Content**: Contact NU scholars
- **Deployment**: Railway/Vercel support

## 🎯 Success Criteria

### Technical Success
- ✅ All endpoints responding correctly
- ✅ Frontend-backend integration working
- ✅ CI/CD pipeline functioning
- ✅ Performance within acceptable limits

### Islamic Content Success
- ✅ Authentic NU methodology implementation
- ✅ Proper source attribution
- ✅ Cultural sensitivity maintained
- ✅ Arabic text properly displayed

### User Experience Success
- ✅ Intuitive chat interface
- ✅ Fast response times
- ✅ Helpful error messages
- ✅ Mobile-friendly design

---

## 📋 Quick Commands Reference

```bash
# Local development
cd backend && uvicorn main:app --reload
cd frontend && npm run dev

# Testing
python test_backend.py
cd frontend && npm run build

# Code quality
black backend/ && isort backend/
cd frontend && npm run lint && npm run type-check

# Deployment
git add . && git commit -m "Deploy update" && git push
```

**🎉 Ready for production when all items are checked!**
