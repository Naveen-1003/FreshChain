"""
Create test data directly via API endpoints
"""
import requests
import json
from datetime import datetime, timedelta
import time

# Backend configuration
BASE_URL = "http://localhost:8000"

def get_auth_token(username, password):
    """Get authentication token"""
    response = requests.post(f"{BASE_URL}/api/auth/login-json", json={
        "username": username,
        "password": password
    })
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        print(f"Authentication failed for {username}: {response.status_code}")
        return None

def create_test_batches():
    """Create test batches via API"""
    print("🚀 Creating test batches...")
    
    # Get manufacturer token
    token = get_auth_token("manufacturer1", "password123")
    if not token:
        return False
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Create test batches
    batches = [
        {
            "product_name": "Organic Apples",
            "quantity": 100,
            "unit": "kg",
            "production_date": (datetime.now() - timedelta(days=3)).isoformat(),
            "expiry_date": (datetime.now() + timedelta(days=10)).isoformat(),
            "min_temperature": 2,
            "max_temperature": 8,
            "min_humidity": 80,
            "max_humidity": 95
        },
        {
            "product_name": "Fresh Bananas", 
            "quantity": 150,
            "unit": "kg",
            "production_date": (datetime.now() - timedelta(days=2)).isoformat(),
            "expiry_date": (datetime.now() + timedelta(days=7)).isoformat(),
            "min_temperature": 12,
            "max_temperature": 15,
            "min_humidity": 85,
            "max_humidity": 95
        }
    ]
    
    created_batches = []
    for batch_data in batches:
        response = requests.post(f"{BASE_URL}/api/batches", 
                               json=batch_data, headers=headers)
        if response.status_code == 200:
            batch_info = response.json()
            created_batches.append(batch_info["batch_id"])
            print(f"✓ Created batch: {batch_info['batch_id']}")
        else:
            print(f"✗ Failed to create batch: {response.status_code} - {response.text}")
    
    return created_batches

def create_test_scans(batch_ids):
    """Create test scan records"""
    print("📱 Creating test scan records...")
    
    if not batch_ids:
        print("No batches available for scanning")
        return False
    
    # Transporter scans
    transporter_token = get_auth_token("transporter1", "password123")
    if transporter_token:
        headers = {"Authorization": f"Bearer {transporter_token}"}
        
        for batch_id in batch_ids:
            scan_data = {
                "batch_id": batch_id,
                "operator_id": "transporter1",
                "operator_type": "transporter",
                "scan_type": "pickup",
                "location": "Warehouse A",
                "timestamp": datetime.now().isoformat(),
                "vehicle_id": "TRUCK_001"
            }
            
            response = requests.post(f"{BASE_URL}/api/qr/scan", 
                                   json=scan_data, headers=headers)
            if response.status_code == 200:
                print(f"✓ Transporter scanned batch: {batch_id}")
            else:
                print(f"✗ Transporter scan failed: {response.status_code} - {response.text}")
            
            time.sleep(1)  # Small delay between scans
    
    # Retailer scans
    retailer_token = get_auth_token("retailer1", "password123") 
    if retailer_token:
        headers = {"Authorization": f"Bearer {retailer_token}"}
        
        for batch_id in batch_ids:
            scan_data = {
                "batch_id": batch_id,
                "operator_id": "retailer1",
                "operator_type": "retailer", 
                "scan_type": "receive",
                "location": "Store A",
                "timestamp": datetime.now().isoformat()
            }
            
            response = requests.post(f"{BASE_URL}/api/qr/scan",
                                   json=scan_data, headers=headers)
            if response.status_code == 200:
                print(f"✓ Retailer scanned batch: {batch_id}")
            else:
                print(f"✗ Retailer scan failed: {response.status_code} - {response.text}")
            
            time.sleep(1)  # Small delay between scans

def main():
    """Main function to create all test data"""
    print("🔧 Setting up role-based test data...")
    print("=" * 50)
    
    # Create batches first
    batch_ids = create_test_batches()
    
    if batch_ids:
        print(f"\n📦 Created {len(batch_ids)} batches")
        
        # Create scan records
        create_test_scans(batch_ids)
        
        print("\n📊 Expected role-based access:")
        print("👨‍🏭 manufacturer1: Should see created batches")
        print("🚛 transporter1: Should see scanned batches")
        print("🏪 retailer1: Should see received batches")
        print("\n✅ Test data setup completed!")
        print("\n🌐 Test the frontend at: http://localhost:3000")
        
    else:
        print("❌ Failed to create test data")

if __name__ == "__main__":
    main()