# ESP32 Sensor Monitoring System

This system provides real-time monitoring of ESP32 sensors for temperature, humidity, and connectivity status.

## Features

- **Real-time Sensor Status**: Monitor ESP32 device connectivity, battery levels, and signal strength
- **Environmental Data Tracking**: Track temperature and humidity readings from sensors
- **Threshold Alerts**: Automatic alerts when environmental parameters exceed safe limits
- **Professional Dashboard**: Clean, minimal UI for monitoring sensor data

## Setup Instructions

### 1. Backend Setup

1. Start the backend server:
   ```bash
   cd "C:\Users\K R ARAVIND\OneDrive\Desktop\freshness"
   python complete_backend.py
   ```

2. The backend will be available at `http://localhost:8000`

### 2. Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd "C:\Users\K R ARAVIND\OneDrive\Desktop\freshness\hyperledger-food-traceability\frontend"
   ```

2. Install dependencies (if not already done):
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open `http://localhost:3000` in your browser

### 3. Login Credentials

Use these test credentials to access the system:

- **Manufacturer**: username: `manufacturer1`, password: `password123`
- **Transporter**: username: `transporter1`, password: `password123`
- **Retailer**: username: `retailer1`, password: `password123`

### 4. ESP32 Sensor Testing

To test the sensor monitoring system without real ESP32 hardware:

1. Run the ESP32 simulator:
   ```bash
   cd "C:\Users\K R ARAVIND\OneDrive\Desktop\freshness"
   python esp32_simulator.py
   ```

2. The simulator will:
   - Send heartbeat signals every 30 seconds
   - Send environmental data every 60 seconds
   - Simulate realistic sensor readings

## Navigation

After logging in, use the sidebar navigation:

- **Dashboard**: Overview of all system components
- **Batches**: Manage product batches
- **Environmental**: View environmental monitoring data
- **Sensors**: 🆕 **ESP32 sensor connectivity and status monitoring**
- **QR Codes**: Generate and manage QR codes
- **Freshness**: AI-powered freshness assessment

## ESP32 Sensor Page Features

The Sensors page (`/sensors`) provides:

### Sensor Status Overview
- Total sensors count
- Online/offline sensor counts
- Recent readings count

### Detailed Sensor Information
- Sensor ID and Vehicle ID
- Connection status (online/offline)
- IP address
- Signal strength with visual indicators
- Battery level with alerts
- Firmware version
- Last seen timestamp

### Recent Environmental Data
- Temperature and humidity readings from last 24 hours
- Threshold violation alerts
- Sensor location data
- Batch and product information

### Real-time Updates
- Auto-refresh every 30 seconds
- Manual refresh button
- Live status indicators

## API Endpoints for ESP32

### Heartbeat Endpoint
```
POST /api/sensors/heartbeat
Content-Type: application/json

{
  "sensor_id": "ESP32_001",
  "vehicle_id": "TRUCK_001",
  "firmware_version": "1.0.0",
  "battery_level": 85,
  "signal_strength": -45
}
```

### Environmental Data Endpoint
```
POST /api/environmental/add
Authorization: Bearer <token>
Content-Type: application/json

{
  "batch_id": "BATCH_1234",
  "temperature": 25.5,
  "humidity": 65.0,
  "location": "GPS:40.7128,-74.0060",
  "sensor_id": "ESP32_001",
  "reading_type": "transport"
}
```

### Status Check Endpoint
```
GET /api/sensors/status
Authorization: Bearer <token>
```

## Temperature and Humidity Thresholds

The system monitors for threshold violations:
- **Temperature**: Alerts if outside safe range for specific products
- **Humidity**: Alerts if humidity levels could damage products
- **Alerts**: Visual indicators on the dashboard when thresholds are exceeded

## Troubleshooting

### Sensors Not Appearing
1. Ensure ESP32 devices are sending heartbeat signals
2. Check network connectivity between ESP32 and backend
3. Verify backend is running on correct port (8000)

### Environmental Data Not Showing
1. Ensure ESP32 is authenticated (has valid token)
2. Check that environmental data POST requests are successful
3. Verify data format matches expected schema

### Connection Issues
1. Check if backend is accessible at `http://localhost:8000`
2. Verify frontend is running on `http://localhost:3000`
3. Check browser console for API errors

## ESP32 Integration Code

For real ESP32 devices, use this sample code structure:

```cpp
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <DHT.h>

// WiFi credentials
const char* ssid = "your_wifi_ssid";
const char* password = "your_wifi_password";

// Backend configuration
const char* backend_url = "http://your-backend-ip:8000";
const char* sensor_id = "ESP32_001";
const char* vehicle_id = "TRUCK_001";

// Sensor configuration
DHT dht(2, DHT22);  // DHT22 sensor on pin 2

void sendHeartbeat() {
  HTTPClient http;
  http.begin(String(backend_url) + "/api/sensors/heartbeat");
  http.addHeader("Content-Type", "application/json");
  
  StaticJsonDocument<200> doc;
  doc["sensor_id"] = sensor_id;
  doc["vehicle_id"] = vehicle_id;
  doc["firmware_version"] = "1.0.0";
  doc["battery_level"] = 100;  // Implement battery reading
  doc["signal_strength"] = WiFi.RSSI();
  
  String json;
  serializeJson(doc, json);
  
  int httpResponseCode = http.POST(json);
  http.end();
}

void sendEnvironmentalData() {
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();
  
  if (isnan(temperature) || isnan(humidity)) {
    return;  // Invalid reading
  }
  
  // First get authentication token
  // Then send environmental data with token
  // Implementation details depend on your authentication flow
}
```

## System Architecture

```
ESP32 Sensors → Backend API → Database
                     ↓
Frontend Dashboard ← API Endpoints
```

The system provides a complete IoT monitoring solution for temperature-sensitive supply chain management.