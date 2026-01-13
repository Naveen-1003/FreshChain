import requests
import json
from pathlib import Path

# Base URL for the API (adjust if running on different host/port)
BASE_URL = "http://localhost:8000"

def test_freshness_evaluation():
    """Test the freshness evaluation endpoint"""
    print("🧪 Testing Freshness Evaluation Endpoint")
    print("-" * 40)
    
    # You'll need to provide a valid image file path
    image_path = input("Enter path to an image file (or press Enter to skip): ").strip()
    
    if not image_path or not Path(image_path).exists():
        print("⚠️ No valid image provided, skipping freshness evaluation test")
        return
    
    try:
        with open(image_path, 'rb') as image_file:
            files = {'file': image_file}
            response = requests.post(f"{BASE_URL}/evaluate-freshness", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"   Prediction Score: {result['prediction_score']:.4f}")
            print(f"   Category: {result['freshness_category']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Message: {result['message']}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   Details: {response.text}")
    
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print()

def test_shelf_life_prediction():
    """Test the shelf life prediction endpoint"""
    print("🧪 Testing Shelf Life Prediction Endpoint")
    print("-" * 40)
    
    # Test data
    test_cases = [
        {"fruit_name": "apple", "storage_temperature": 5.0},
        {"fruit_name": "banana", "storage_temperature": 25.0},
        {"fruit_name": "tomato", "storage_temperature": 15.0}
    ]
    
    for case in test_cases:
        try:
            response = requests.post(
                f"{BASE_URL}/predict-shelf-life",
                json=case,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {case['fruit_name'].capitalize()} at {case['storage_temperature']}°C:")
                print(f"   Shelf Life Ratio: {result['shelf_life_ratio']:.2f}")
                print(f"   Conclusion: {result['conclusion']}")
            else:
                print(f"❌ Error for {case['fruit_name']}: {response.status_code}")
                print(f"   Details: {response.text}")
        
        except Exception as e:
            print(f"❌ Exception for {case['fruit_name']}: {e}")
    
    print()

def test_available_items():
    """Test the available items endpoint"""
    print("🧪 Testing Available Items Endpoint")
    print("-" * 40)
    
    try:
        response = requests.get(f"{BASE_URL}/available-items")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Available Items:")
            for item, details in result['available_items'].items():
                print(f"   {item}: {details['metric']}")
            print(f"   Total: {result['total_count']} items")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   Details: {response.text}")
    
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print()

def test_health_check():
    """Test the health check endpoint"""
    print("🧪 Testing Health Check Endpoint")
    print("-" * 40)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Health Check:")
            print(f"   Status: {result['status']}")
            print(f"   Model Status: {result['model_status']}")
            print(f"   Available Items: {result['available_items_count']}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   Details: {response.text}")
    
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print()

def main():
    """Run all tests"""
    print("🚀 FastAPI Fruit & Vegetable Freshness API Test Client")
    print("=" * 60)
    
    # Check if API is running
    try:
        response = requests.get(BASE_URL)
        if response.status_code != 200:
            print(f"❌ API not accessible at {BASE_URL}")
            print("   Make sure the FastAPI server is running with: python main.py")
            return
    except requests.ConnectionError:
        print(f"❌ Cannot connect to API at {BASE_URL}")
        print("   Make sure the FastAPI server is running with: python main.py")
        return
    
    print(f"✅ API is accessible at {BASE_URL}")
    print()
    
    # Run tests
    test_health_check()
    test_available_items()
    test_shelf_life_prediction()
    test_freshness_evaluation()
    
    print("🎉 All tests completed!")

if __name__ == "__main__":
    main()