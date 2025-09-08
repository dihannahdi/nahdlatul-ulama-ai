#!/bin/bash
# Railway startup script for Nahdlatul Ulama AI

echo "🚀 Starting Nahdlatul Ulama AI on Railway..."
echo "🌍 Environment: ${RAILWAY_ENVIRONMENT_NAME:-development}"
echo "🔗 Port: ${PORT:-8000}"

# Set production environment
export PRODUCTION=true

# Start the application
exec uvicorn full_production_main:production_app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
