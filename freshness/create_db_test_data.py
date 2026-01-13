"""
Direct database insertion for test data
This script connects directly to PostgreSQL and inserts test data
"""

import psycopg2
from datetime import datetime, timedelta
import uuid
import json

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'visaion',
    'user': 'postgres',
    'password': 'password',  # Based on backend configuration
    'port': 5432
}

def create_test_data():
    """Create test data directly in database"""
    try:
        # Connect to database
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        print("🔗 Connected to database")
        
        # Create test batches
        print("📦 Creating test batches...")
        
        test_batches = [
            {
                'batch_id': 'BATCH_MFG1_001',
                'product_name': 'Organic Apples',
                'quantity': 100.0,
                'harvest_date': datetime.now() - timedelta(days=3),
                'origin': 'Farm A',
                'manufacturer_id': 'manufacturer1',
                'current_location': 'Warehouse A',
                'status': 'In Transit'
            },
            {
                'batch_id': 'BATCH_MFG1_002',
                'product_name': 'Fresh Bananas',
                'quantity': 150.0,
                'harvest_date': datetime.now() - timedelta(days=2),
                'origin': 'Farm B',
                'manufacturer_id': 'manufacturer1',
                'current_location': 'Distribution Center',
                'status': 'Delivered'
            }
        ]
        
        for batch in test_batches:
            cur.execute("""
                INSERT INTO batches 
                (batch_id, product_name, quantity, harvest_date, origin, manufacturer_id, current_location, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (batch_id) DO UPDATE SET
                product_name = EXCLUDED.product_name,
                manufacturer_id = EXCLUDED.manufacturer_id,
                current_location = EXCLUDED.current_location,
                status = EXCLUDED.status
            """, (batch['batch_id'], batch['product_name'], batch['quantity'], 
                  batch['harvest_date'], batch['origin'], batch['manufacturer_id'],
                  batch['current_location'], batch['status']))
            
            print(f"✓ Created/Updated batch: {batch['batch_id']}")
        
        # Create test QR scans using the existing qr_scans table
        print("📱 Creating test scan records...")
        
        # First get the batch IDs (database internal IDs)
        cur.execute("SELECT id, batch_id FROM batches WHERE batch_id IN %s", 
                   (('BATCH_MFG1_001', 'BATCH_MFG1_002'),))
        batch_mappings = {row[1]: row[0] for row in cur.fetchall()}
        
        test_scans = [
            # Transporter scans
            {
                'batch_id': batch_mappings.get('BATCH_MFG1_001'),
                'scanned_by': 'transporter1',
                'scan_type': 'pickup',
                'location': 'Warehouse A',
                'additional_data': json.dumps({
                    'operator_type': 'transporter',
                    'vehicle_id': 'TRUCK_001',
                    'scan_id': str(uuid.uuid4())
                })
            },
            {
                'batch_id': batch_mappings.get('BATCH_MFG1_002'),
                'scanned_by': 'transporter1',
                'scan_type': 'pickup',
                'location': 'Warehouse A',
                'additional_data': json.dumps({
                    'operator_type': 'transporter',
                    'vehicle_id': 'TRUCK_001',
                    'scan_id': str(uuid.uuid4())
                })
            },
            # Retailer scans
            {
                'batch_id': batch_mappings.get('BATCH_MFG1_001'),
                'scanned_by': 'retailer1',
                'scan_type': 'receive',
                'location': 'Store A',
                'additional_data': json.dumps({
                    'operator_type': 'retailer',
                    'scan_id': str(uuid.uuid4())
                })
            },
            {
                'batch_id': batch_mappings.get('BATCH_MFG1_002'),
                'scanned_by': 'retailer1',
                'scan_type': 'receive',
                'location': 'Store A',
                'additional_data': json.dumps({
                    'operator_type': 'retailer',
                    'scan_id': str(uuid.uuid4())
                })
            }
        ]
        
        for scan in test_scans:
            if scan['batch_id']:  # Only insert if batch exists
                cur.execute("""
                    INSERT INTO qr_scans 
                    (batch_id, scanned_by, scan_type, location, additional_data, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (scan['batch_id'], scan['scanned_by'], scan['scan_type'],
                      scan['location'], scan['additional_data'], datetime.now()))
                
                print(f"✓ Created scan: {scan['scanned_by']} -> {scan['scan_type']}")
        
        # Create some environmental data
        print("🌡️ Creating environmental data...")
        
        env_data = [
            {
                'reading_id': str(uuid.uuid4()),
                'vehicle_id': 'TRUCK_001',
                'batch_ids': ['BATCH_MFG1_001'],
                'batch_id': 'BATCH_MFG1_001',
                'temperature': 23.5,
                'humidity': 67.2,
                'timestamp': datetime.now() - timedelta(hours=2),
                'location': 'In Transit',
                'sensor_id': 'ESP32_001',
                'reading_type': 'transport'
            },
            {
                'reading_id': str(uuid.uuid4()),
                'vehicle_id': 'TRUCK_001',
                'batch_ids': ['BATCH_MFG1_002'],
                'batch_id': 'BATCH_MFG1_002',
                'temperature': 25.1,
                'humidity': 72.8,
                'timestamp': datetime.now() - timedelta(hours=1),
                'location': 'Store A',
                'sensor_id': 'ESP32_001',
                'reading_type': 'storage'
            }
        ]
        
        for env in env_data:
            cur.execute("""
                INSERT INTO environmental_readings 
                (reading_id, vehicle_id, batch_ids, batch_id, temperature, humidity, timestamp, location, sensor_id, reading_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (env['reading_id'], env['vehicle_id'], env['batch_ids'],
                  env['batch_id'], env['temperature'], env['humidity'], 
                  env['timestamp'], env['location'], env['sensor_id'], env['reading_type']))
            
            print(f"✓ Created environmental reading: T={env['temperature']}°C, H={env['humidity']}%")
        
        # Commit all changes
        conn.commit()
        
        print("\n📊 Expected role-based access:")
        print("👨‍🏭 manufacturer1: Should see BATCH_MFG1_001, BATCH_MFG1_002")
        print("🚛 transporter1: Should see BATCH_MFG1_001, BATCH_MFG1_002 (scanned)")
        print("🏪 retailer1: Should see BATCH_MFG1_001, BATCH_MFG1_002 (received)")
        print("\n✅ Test data creation completed!")
        
        # Verify the data
        print("\n🔍 Verification:")
        
        # Check batches
        cur.execute("SELECT batch_id, product_name, manufacturer_id FROM batches")
        batches = cur.fetchall()
        print(f"Total batches in DB: {len(batches)}")
        for batch in batches:
            print(f"  - {batch[0]}: {batch[1]} (by {batch[2]})")
        
        # Check scans
        cur.execute("""
            SELECT qs.scanned_by, qs.scan_type, qs.additional_data, b.batch_id
            FROM qr_scans qs
            JOIN batches b ON qs.batch_id = b.id
        """)
        scans = cur.fetchall()
        print(f"Total scans in DB: {len(scans)}")
        for scan in scans:
            operator_type = json.loads(scan[2]).get('operator_type', 'unknown')
            print(f"  - {scan[0]} ({operator_type}) {scan[1]} batch {scan[3]}")
        
    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()
            print("🔗 Database connection closed")

if __name__ == "__main__":
    create_test_data()