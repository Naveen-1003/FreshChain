import requests
import json

# Test the sensors status endpoint directly
url = "http://localhost:8000/api/sensors/status"
headers = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJyZXRhaWxlcjEiLCJyb2xlIjoicmV0YWlsZXIiLCJleHAiOjE3MzQxODUwNTZ9.mHRv7OS5M1TSGVoyakl-22Xd5NjMr4Bv1aR2m-BypOA"
}

try:
    print("🔍 Testing sensors endpoint...")
    response = requests.get(url, headers=headers)
    print(f"📊 Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("📦 Response Data:")
        print(json.dumps(data, indent=2, default=str))
        
        # Check for sensors
        if 'sensors' in data and data['sensors']:
            print(f"\n🔗 Found {len(data['sensors'])} sensors:")
            for sensor in data['sensors']:
                print(f"  Sensor ID: {sensor.get('sensor_id')}")
                print(f"  Status: {sensor.get('status')}")
                print(f"  Last Temperature: {sensor.get('last_temperature')}°C")
                print(f"  Last Humidity: {sensor.get('last_humidity')}%")
                print(f"  Temperature: {sensor.get('temperature')}°C")
                print(f"  Humidity: {sensor.get('humidity')}%")
                print("  ---")
        else:
            print("❌ No sensors found in response")
    else:
        print(f"❌ Error: {response.text}")
        
except Exception as e:
    print(f"❌ Exception: {e}")