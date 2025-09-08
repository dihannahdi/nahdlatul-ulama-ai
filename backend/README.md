# Nahdlatul Ulama AI - Backend

FastAPI backend service for Islamic jurisprudence AI using NU methodology.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Get free API keys:**
   - **Groq**: https://console.groq.com (15 requests/minute free)
   - **Google Gemini**: https://ai.google.dev (Free tier available)

4. **Run the server:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## 📡 API Endpoints

### Main Endpoints
- `POST /ask` - Ask Islamic jurisprudence questions
- `GET /methods` - Get available reasoning methods  
- `GET /search` - Search Islamic texts
- `GET /stats` - Get system statistics

### Health Check
- `GET /` - Basic health check
- `GET /health` - Detailed health status

## 🔧 Islamic Reasoning Methods

1. **Bayani (البياني)** - Textual analysis from Quran & Sunnah
2. **Qiyasi (القياسي)** - Analogical reasoning
3. **Istishlahi (الاستصلاحي)** - Public interest consideration
4. **Maqashidi (المقاصدي)** - Objectives of Sharia

## 💾 Data Sources

- **References**: NU methodology texts in `/references`
- **Knowledge Base**: Islamic texts in `/sql_chunks`
- **Vector Store**: ChromaDB for semantic search

## 🏛️ NU Principles (Fikrah Nahdliyah)

- **Tawassuth (التوسط)**: Moderation
- **Tasamuh (التسامح)**: Tolerance  
- **Tawazun (التوازن)**: Balance
- **I'tidal (الاعتدال)**: Justice

## 🔍 Example Request

```json
{
  "question": "Bagaimana hukum menggunakan teknologi AI dalam pendidikan Islam?",
  "method": "istishlahi",
  "language": "id"
}
```

## 🌐 Deployment

### Railway Deployment
1. Connect GitHub repository to Railway
2. Set environment variables in Railway dashboard
3. Deploy automatically on push

### Environment Variables
```bash
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
```

## 📊 System Requirements

- **Python**: 3.9+
- **Memory**: 2GB+ (for embeddings)
- **Storage**: 1GB+ (for vector database)

## 🧪 Testing

```bash
# Run health check
curl http://localhost:8000/health

# Test question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Apa itu fiqh?"}'
```

## 📈 Performance

- **Cold start**: ~30 seconds (loading embeddings)
- **Response time**: 2-5 seconds per question
- **Throughput**: Depends on LLM API limits

## 🆓 Free Tier Limits

- **Groq**: 15 requests/minute
- **Gemini**: 60 requests/minute  
- **Railway**: 500 hours/month
- **ChromaDB**: No limits (self-hosted)

## 🔧 Troubleshooting

### Common Issues

1. **Import errors**: Install requirements.txt
2. **API key errors**: Set valid API keys in .env
3. **Memory errors**: Reduce batch size in data processor
4. **Slow startup**: Normal for first run (building embeddings)

### Logs

Check Railway logs or local console for detailed error messages.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly  
5. Submit pull request

## 📄 License

MIT License - Free for educational and religious purposes.
