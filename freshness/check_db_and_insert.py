"""
Simple test data insertion using SQL commands
"""

import psycopg2
from datetime import datetime
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

def check_and_create_data():
    """Check table structure and create test data"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        print("🔗 Connected to database")
        
        # Check if tables exist and their structure
        print("🔍 Checking table structure...")
        
        # Check batches table
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'batches'
            ORDER BY ordinal_position
        """)
        batch_columns = cur.fetchall()
        print(f"Batches table columns: {batch_columns}")
        
        # Check qr_scans table
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'qr_scans'
            ORDER BY ordinal_position
        """)
        scan_columns = cur.fetchall()
        print(f"QR scans table columns: {scan_columns}")
        
        # If batches table has the right structure, insert data
        batch_column_names = [col[0] for col in batch_columns]
        
        if 'batch_id' in batch_column_names or 'batch_number' in batch_column_names:
            print("✅ Batches table found, inserting test data...")
            
            # Use whatever primary key column exists
            pk_column = 'batch_id' if 'batch_id' in batch_column_names else 'batch_number'
            
            # Insert test batches
            batch_data = [
                ('BATCH_MFG1_001', 'Organic Apples', 100.0, datetime.now(), 'Farm A', 'manufacturer1'),
                ('BATCH_MFG1_002', 'Fresh Bananas', 150.0, datetime.now(), 'Farm B', 'manufacturer1'),
            ]
            
            for batch in batch_data:
                try:
                    cur.execute(f"""
                        INSERT INTO batches 
                        ({pk_column}, product_name, quantity, harvest_date, origin, manufacturer_id)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT ({pk_column}) DO NOTHING
                    """, batch)
                    print(f"✓ Inserted batch: {batch[0]}")
                except Exception as e:
                    print(f"⚠️ Batch insert failed: {e}")
            
            # Get batch IDs for scans
            if 'id' in batch_column_names:
                cur.execute(f"SELECT id, {pk_column} FROM batches WHERE {pk_column} IN %s", 
                           (('BATCH_MFG1_001', 'BATCH_MFG1_002'),))
                batch_mappings = {row[1]: row[0] for row in cur.fetchall()}
                
                # Insert scan records if qr_scans table exists
                if scan_columns:
                    print("📱 Inserting scan records...")
                    
                    scan_data = [
                        (batch_mappings.get('BATCH_MFG1_001'), 'transporter1', 'pickup', 'Warehouse A'),
                        (batch_mappings.get('BATCH_MFG1_001'), 'retailer1', 'receive', 'Store A'),
                        (batch_mappings.get('BATCH_MFG1_002'), 'transporter1', 'pickup', 'Warehouse A'),
                        (batch_mappings.get('BATCH_MFG1_002'), 'retailer1', 'receive', 'Store A'),
                    ]
                    
                    for scan in scan_data:
                        if scan[0]:  # Only if batch exists
                            try:
                                additional_data = json.dumps({
                                    'operator_type': 'transporter' if 'transporter' in scan[1] else 'retailer',
                                    'scan_id': str(uuid.uuid4())
                                })
                                
                                cur.execute("""
                                    INSERT INTO qr_scans 
                                    (batch_id, scanned_by, scan_type, location, additional_data, created_at)
                                    VALUES (%s, %s, %s, %s, %s, %s)
                                """, (scan[0], scan[1], scan[2], scan[3], additional_data, datetime.now()))
                                
                                print(f"✓ Inserted scan: {scan[1]} -> {scan[2]}")
                            except Exception as e:
                                print(f"⚠️ Scan insert failed: {e}")
        
        # Commit changes
        conn.commit()
        print("✅ Data insertion completed!")
        
        # Show final counts
        cur.execute("SELECT COUNT(*) FROM batches")
        batch_count = cur.fetchone()[0]
        print(f"📦 Total batches: {batch_count}")
        
        if scan_columns:
            cur.execute("SELECT COUNT(*) FROM qr_scans")
            scan_count = cur.fetchone()[0]
            print(f"📱 Total scans: {scan_count}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()
            print("🔗 Connection closed")

if __name__ == "__main__":
    check_and_create_data()