import requests
import json

# Test the AI suggestions endpoint
url = "http://localhost:8000/api/ai/suggestions"
headers = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJyZXRhaWxlcjEiLCJyb2xlIjoicmV0YWlsZXIiLCJleHAiOjE3MzQxODUwNTZ9.mHRv7OS5M1TSGVoyakl-22Xd5NjMr4Bv1aR2m-BypOA"
}

try:
    print("🔍 Testing AI suggestions endpoint...")
    response = requests.get(url, headers=headers)
    print(f"📊 Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("📦 Response Data:")
        print(json.dumps(data, indent=2))
        
        print("\n🧠 AI Suggestions:")
        for i, suggestion in enumerate(data.get('suggestions', []), 1):
            print(f"{i}. {suggestion}")
            
    else:
        print(f"❌ Error: {response.text}")
        
except Exception as e:
    print(f"❌ Exception: {e}")