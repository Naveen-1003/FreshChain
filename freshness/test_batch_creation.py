#!/usr/bin/env python3
"""
Test script to verify batch creation functionality works correctly
"""
import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_login():
    """Test login and return token"""
    # OAuth2 form data format
    login_data = {
        "grant_type": "password",  # Required for OAuth2 password flow
        "username": "manufacturer1",
        "password": "password123",
        "scope": "",  # Optional for this implementation
        "client_id": "",  # Optional for this implementation
        "client_secret": ""  # Optional for this implementation
    }
    
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    print("Login request:", login_data)
    response = requests.post(
        f"{BASE_URL}/api/auth/login",
        data=login_data,  # Use data= for form data
        headers=headers
    )
    print(f"Login response status: {response.status_code}")
    print(f"Response headers: {dict(response.headers)}")
    
    try:
        data = response.json()
        print(f"Response data: {json.dumps(data, indent=2)}")
        if response.status_code == 200:
            print(f"Login successful! Token received.")
            return data["access_token"]
        else:
            print(f"Login failed: {data}")
            return None
    except json.JSONDecodeError:
        print(f"Failed to decode JSON response: {response.text}")
        return None

def test_batch_creation(token):
    """Test batch creation with authentication"""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Current date for harvest date
    batch_data = {
        "product_name": "Fresh Tomatoes",
        "quantity": 150.5,
        "harvest_date": datetime.now().strftime('%Y-%m-%d'),  # Changed to match expected format
        "origin": "Green Valley Farm - California",
        "quality_metrics": {
            "freshness_score": 95,
            "organic": True,
            "pesticide_free": True
        }
    }
    
    print("Request data:", json.dumps(batch_data, indent=2))
    response = requests.post(f"{BASE_URL}/api/batches", json=batch_data, headers=headers)
    print(f"Batch creation response status: {response.status_code}")
    print(f"Response headers: {dict(response.headers)}")
    
    try:
        response_data = response.json()
        print(f"Response data: {json.dumps(response_data, indent=2)}")
        if response.status_code == 200:
            print(f"Batch created successfully!")
            return response_data.get("batch_id")
        else:
            print(f"Batch creation failed: {response_data}")
            return None
    except json.JSONDecodeError:
        print(f"Failed to decode JSON response: {response.text}")
        return None

def test_manufacturer_dashboard(token):
    """Test manufacturer dashboard data retrieval"""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Get manufacturer info first
    print("Getting user info...")
    response = requests.get(f"{BASE_URL}/api/auth/me", headers=headers)
    print(f"User info response status: {response.status_code}")
    
    try:
        response_data = response.json()
        print(f"User info response: {json.dumps(response_data, indent=2)}")
        
        if response.status_code == 200:
            company_name = response_data["company_name"]
            print(f"Manufacturer: {company_name}")
            
            # Test dashboard endpoint
            dashboard_url = f"{BASE_URL}/api/manufacturer/{company_name}/dashboard"
            print(f"Getting dashboard from: {dashboard_url}")
            
            dashboard_response = requests.get(dashboard_url, headers=headers)
            print(f"Dashboard response status: {dashboard_response.status_code}")
            
            try:
                dashboard_data = dashboard_response.json()
                print(f"Dashboard response: {json.dumps(dashboard_data, indent=2)}")
                
                if dashboard_response.status_code == 200:
                    print(f"Dashboard data retrieved successfully!")
                    print(f"Total batches: {dashboard_data.get('total_batches', 'N/A')}")
                    print(f"Recent batches count: {len(dashboard_data.get('recent_batches', []))}")
                    return dashboard_data
                else:
                    print(f"Dashboard request failed: {dashboard_data}")
            except json.JSONDecodeError:
                print(f"Failed to decode dashboard response: {dashboard_response.text}")
        else:
            print(f"User info request failed: {response_data}")
    except json.JSONDecodeError:
        print(f"Failed to decode user info response: {response.text}")
    
    return None

def main():
    print("🧪 Testing Batch Creation Functionality\n")
    
    # Test 1: Login
    print("1. Testing login...")
    token = test_login()
    if not token:
        print("❌ Login failed - stopping tests")
        return
    
    print("✅ Login successful\n")
    
    # Test 2: Create batch
    print("2. Testing batch creation...")
    batch_id = test_batch_creation(token)
    if not batch_id:
        print("❌ Batch creation failed")
    else:
        print("✅ Batch creation successful\n")
    
    # Test 3: Check dashboard
    print("3. Testing manufacturer dashboard...")
    dashboard_data = test_manufacturer_dashboard(token)
    if dashboard_data:
        print("✅ Dashboard data retrieved successfully")
    else:
        print("❌ Dashboard data retrieval failed")
    
    print("\n🎉 Test completed!")

if __name__ == "__main__":
    main()