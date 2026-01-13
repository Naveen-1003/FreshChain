"""
ESP32 Sensor Simulator Script
This script simulates ESP32 sensor data and heartbeat signals for testing.
Run this to test the sensor connectivity monitoring system.
"""

import requests
import json
import time
import random
from datetime import datetime

# Backend configuration
BACKEND_URL = "http://localhost:8000"  # Change this to your backend URL
SENSOR_ID = "ESP32_001"
VEHICLE_ID = "TRUCK_001"
BATCH_ID = "BATCH_MFG1_001"  # Use valid test batch

# Sensor configuration
FIRMWARE_VERSION = "1.0.0"
BATTERY_LEVEL = 100
SIGNAL_STRENGTH = -45

def send_heartbeat():
    """Send heartbeat signal to backend"""
    url = f"{BACKEND_URL}/api/sensors/heartbeat"
    
    # Simulate realistic sensor readings in heartbeat
    temperature = round(random.uniform(15.0, 35.0), 1)
    humidity = round(random.uniform(30.0, 90.0), 1)
    
    data = {
        "sensor_id": SENSOR_ID,
        "vehicle_id": VEHICLE_ID,
        "firmware_version": FIRMWARE_VERSION,
        "battery_level": max(20, BATTERY_LEVEL - random.randint(0, 5)),  # Simulate battery drain
        "signal_strength": SIGNAL_STRENGTH + random.randint(-10, 10),  # Simulate signal fluctuation
        "temperature": temperature,  # Include current readings
        "humidity": humidity
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print(f"✓ Heartbeat sent: T={temperature}°C, H={humidity}% at {datetime.now().strftime('%H:%M:%S')}")
            return True
        else:
            print(f"✗ Heartbeat failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ Heartbeat error: {e}")
        return False

def send_environmental_data():
    """Send environmental sensor data to backend"""
    url = f"{BACKEND_URL}/api/environmental/add"
    
    # Simulate realistic temperature and humidity readings
    temperature = round(random.uniform(15.0, 35.0), 1)  # 15-35°C
    humidity = round(random.uniform(30.0, 90.0), 1)     # 30-90% RH
    
    data = {
        "batch_id": BATCH_ID,  # Use the valid test batch
        "temperature": temperature,
        "humidity": humidity,
        "location": f"GPS:{random.uniform(-90, 90):.6f},{random.uniform(-180, 180):.6f}",
        "sensor_id": SENSOR_ID,
        "reading_type": "transport"
    }
    
    try:
        # Get a token first (using test credentials)
        auth_url = f"{BACKEND_URL}/api/auth/login-json"
        auth_response = requests.post(auth_url, json={
            "username": "manufacturer1",
            "password": "password123"
        })
        
        if auth_response.status_code == 200:
            token = auth_response.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            
            response = requests.post(url, json=data, headers=headers)
            if response.status_code == 200:
                print(f"✓ Environmental data sent: T={temperature}°C, H={humidity}%")
                return True
            else:
                print(f"✗ Environmental data failed: {response.status_code} - {response.text}")
                return False
        else:
            print(f"✗ Authentication failed: {auth_response.status_code} - {auth_response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Environmental data error: {e}")
        return False

def main():
    """Main simulation loop"""
    print("🚀 Starting ESP32 Sensor Simulator")
    print(f"📡 Sensor ID: {SENSOR_ID}")
    print(f"🚛 Vehicle ID: {VEHICLE_ID}")
    print(f"🌐 Backend URL: {BACKEND_URL}")
    print("=" * 50)
    
    heartbeat_counter = 0
    data_counter = 0
    
    try:
        while True:
            # Send heartbeat every 30 seconds
            if heartbeat_counter % 6 == 0:  # Every 6 iterations (30 seconds at 5-second intervals)
                if send_heartbeat():
                    heartbeat_counter = 0
            
            # Send environmental data every 60 seconds
            if data_counter % 12 == 0:  # Every 12 iterations (60 seconds at 5-second intervals)
                if send_environmental_data():
                    data_counter = 0
            
            heartbeat_counter += 1
            data_counter += 1
            
            # Wait 5 seconds before next iteration
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n🛑 Simulator stopped by user")
    except Exception as e:
        print(f"\n💥 Simulator error: {e}")

if __name__ == "__main__":
    main()