# Quick Test Script for Nahdlatul Ulama AI

import requests
import json
import time

def test_backend_health():
    """Test if backend is running and healthy"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend health check: PASSED")
            return True
        else:
            print(f"❌ Backend health check: FAILED (Status: {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Backend health check: FAILED (Error: {e})")
        return False

def test_methods_endpoint():
    """Test the methods endpoint"""
    try:
        response = requests.get("http://localhost:8000/methods", timeout=5)
        if response.status_code == 200:
            methods = response.json()
            print("✅ Methods endpoint: PASSED")
            print(f"   Available methods: {', '.join(methods)}")
            return True
        else:
            print(f"❌ Methods endpoint: FAILED (Status: {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Methods endpoint: FAILED (Error: {e})")
        return False

def test_search_endpoint():
    """Test the search endpoint"""
    try:
        payload = {
            "query": "prayer",
            "limit": 3
        }
        response = requests.post(
            "http://localhost:8000/search", 
            json=payload, 
            timeout=10
        )
        if response.status_code == 200:
            results = response.json()
            print("✅ Search endpoint: PASSED")
            print(f"   Found {len(results.get('results', []))} results")
            return True
        else:
            print(f"❌ Search endpoint: FAILED (Status: {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Search endpoint: FAILED (Error: {e})")
        return False

def test_ask_endpoint():
    """Test the ask endpoint with a simple Islamic question"""
    try:
        payload = {
            "question": "What are the conditions for valid prayer?",
            "method": "bayani",
            "context": "General Islamic jurisprudence"
        }
        print("🤔 Testing ask endpoint (this may take a few seconds)...")
        response = requests.post(
            "http://localhost:8000/ask", 
            json=payload, 
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            print("✅ Ask endpoint: PASSED")
            print(f"   Answer length: {len(result.get('answer', ''))} characters")
            print(f"   Sources found: {len(result.get('sources', []))}")
            return True
        else:
            print(f"❌ Ask endpoint: FAILED (Status: {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Ask endpoint: FAILED (Error: {e})")
        return False

def test_stats_endpoint():
    """Test the stats endpoint"""
    try:
        response = requests.get("http://localhost:8000/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print("✅ Stats endpoint: PASSED")
            print(f"   Documents in database: {stats.get('total_documents', 'Unknown')}")
            return True
        else:
            print(f"❌ Stats endpoint: FAILED (Status: {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Stats endpoint: FAILED (Error: {e})")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Nahdlatul Ulama AI Backend")
    print("=" * 50)
    
    # Track test results
    tests = [
        ("Backend Health", test_backend_health),
        ("Methods Endpoint", test_methods_endpoint),
        ("Search Endpoint", test_search_endpoint),
        ("Stats Endpoint", test_stats_endpoint),
        ("Ask Endpoint", test_ask_endpoint),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📝 Running: {test_name}")
        if test_func():
            passed += 1
        time.sleep(1)  # Brief pause between tests
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your Nahdlatul Ulama AI backend is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the backend logs and configuration.")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure the backend is running: uvicorn main:app --reload")
        print("2. Check environment variables (.env file)")
        print("3. Verify API keys are valid")
        print("4. Ensure ChromaDB is initialized with data")

if __name__ == "__main__":
    main()
