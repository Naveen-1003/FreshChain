"""
Insert test data using correct table structure
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
    'password': 'password',
    'port': 5432
}

def create_correct_test_data():
    """Create test data using the actual table structure"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        print("🔗 Connected to database")
        print("📦 Creating test batches with correct structure...")
        
        # Insert test batches using actual column names
        batch_data = [
            {
                'id': str(uuid.uuid4()),
                'batch_number': 'BATCH_MFG1_001',
                'product_name': 'Organic Apples',
                'quantity': 100,
                'manufacturer': 'manufacturer1',  # This maps to manufacturer_id in filtering
                'production_date': datetime.now() - timedelta(days=3),
                'expiry_date': datetime.now() + timedelta(days=10),
                'current_location': 'Warehouse A',
                'status': 'In Transit',
                'unit': 'kg'
            },
            {
                'id': str(uuid.uuid4()),
                'batch_number': 'BATCH_MFG1_002',
                'product_name': 'Fresh Bananas',
                'quantity': 150,
                'manufacturer': 'manufacturer1',
                'production_date': datetime.now() - timedelta(days=2),
                'expiry_date': datetime.now() + timedelta(days=7),
                'current_location': 'Distribution Center',
                'status': 'Delivered',
                'unit': 'kg'
            }
        ]
        
        batch_ids = {}  # Store mapping of batch_number to UUID
        
        for batch in batch_data:
            try:
                cur.execute("""
                    INSERT INTO batches 
                    (id, batch_number, product_name, quantity, manufacturer, production_date, expiry_date, current_location, status, unit)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (batch_number) DO UPDATE SET
                    product_name = EXCLUDED.product_name,
                    manufacturer = EXCLUDED.manufacturer,
                    current_location = EXCLUDED.current_location,
                    status = EXCLUDED.status
                """, (batch['id'], batch['batch_number'], batch['product_name'], 
                      batch['quantity'], batch['manufacturer'], batch['production_date'],
                      batch['expiry_date'], batch['current_location'], batch['status'], batch['unit']))
                
                batch_ids[batch['batch_number']] = batch['id']
                print(f"✓ Created batch: {batch['batch_number']} ({batch['product_name']})")
                
            except Exception as e:
                print(f"⚠️ Batch insert failed: {e}")
                conn.rollback()
                break
        
        if batch_ids:
            print("📱 Creating scan records...")
            
            # Create scan records using correct table structure
            scan_data = [
                # Transporter scans
                {
                    'id': str(uuid.uuid4()),
                    'batch_id': batch_ids.get('BATCH_MFG1_001'),
                    'scanned_by': 'transporter1',
                    'scan_type': 'pickup',
                    'scan_timestamp': datetime.now() - timedelta(hours=6),
                    'location': 'Warehouse A',
                    'additional_data': json.dumps({
                        'operator_type': 'transporter',
                        'vehicle_id': 'TRUCK_001'
                    })
                },
                {
                    'id': str(uuid.uuid4()),
                    'batch_id': batch_ids.get('BATCH_MFG1_002'),
                    'scanned_by': 'transporter1',
                    'scan_type': 'pickup',
                    'scan_timestamp': datetime.now() - timedelta(hours=4),
                    'location': 'Warehouse A',
                    'additional_data': json.dumps({
                        'operator_type': 'transporter',
                        'vehicle_id': 'TRUCK_001'
                    })
                },
                # Retailer scans
                {
                    'id': str(uuid.uuid4()),
                    'batch_id': batch_ids.get('BATCH_MFG1_001'),
                    'scanned_by': 'retailer1',
                    'scan_type': 'receive',
                    'scan_timestamp': datetime.now() - timedelta(hours=2),
                    'location': 'Store A',
                    'additional_data': json.dumps({
                        'operator_type': 'retailer'
                    })
                },
                {
                    'id': str(uuid.uuid4()),
                    'batch_id': batch_ids.get('BATCH_MFG1_002'),
                    'scanned_by': 'retailer1',
                    'scan_type': 'receive',
                    'scan_timestamp': datetime.now() - timedelta(hours=1),
                    'location': 'Store A',
                    'additional_data': json.dumps({
                        'operator_type': 'retailer'
                    })
                }
            ]
            
            for scan in scan_data:
                if scan['batch_id']:
                    try:
                        cur.execute("""
                            INSERT INTO qr_scans 
                            (id, batch_id, scanned_by, scan_type, scan_timestamp, location, additional_data)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, (scan['id'], scan['batch_id'], scan['scanned_by'], 
                              scan['scan_type'], scan['scan_timestamp'], scan['location'],
                              scan['additional_data']))
                        
                        operator_type = json.loads(scan['additional_data']).get('operator_type')
                        print(f"✓ Created scan: {scan['scanned_by']} ({operator_type}) -> {scan['scan_type']}")
                        
                    except Exception as e:
                        print(f"⚠️ Scan insert failed: {e}")
            
        # Commit all changes
        conn.commit()
        print("✅ Test data creation completed!")
        
        # Verify the data
        print("\n🔍 Verification:")
        cur.execute("SELECT batch_number, product_name, manufacturer FROM batches")
        batches = cur.fetchall()
        print(f"📦 Total batches: {len(batches)}")
        for batch in batches:
            print(f"  - {batch[0]}: {batch[1]} (by {batch[2]})")
        
        cur.execute("""
            SELECT qs.scanned_by, qs.scan_type, qs.additional_data::text, b.batch_number
            FROM qr_scans qs
            JOIN batches b ON qs.batch_id = b.id
        """)
        scans = cur.fetchall()
        print(f"📱 Total scans: {len(scans)}")
        for scan in scans:
            try:
                operator_type = json.loads(scan[2]).get('operator_type', 'unknown')
                print(f"  - {scan[0]} ({operator_type}) {scan[1]} batch {scan[3]}")
            except:
                print(f"  - {scan[0]} {scan[1]} batch {scan[3]}")
        
        print("\n📊 Expected role-based access:")
        print("👨‍🏭 manufacturer1: Should see BATCH_MFG1_001, BATCH_MFG1_002")
        print("🚛 transporter1: Should see BATCH_MFG1_001, BATCH_MFG1_002 (scanned)")
        print("🏪 retailer1: Should see BATCH_MFG1_001, BATCH_MFG1_002 (received)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'conn' in locals():
            conn.close()
            print("🔗 Connection closed")

if __name__ == "__main__":
    create_correct_test_data()