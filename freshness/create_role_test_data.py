"""
Create test data for role-based filtering
This script creates batches and scan records to demonstrate role-based access control
"""

import asyncio
import asyncpg
import uuid
from datetime import datetime, timedelta
import json

# Database configuration
DATABASE_URL = "postgresql://postgres:admin@localhost/food_traceability"

async def create_test_data():
    """Create test batches and scan records for role-based filtering"""
    conn = await asyncpg.connect(DATABASE_URL)
    
    try:
        print("🚀 Creating test data for role-based filtering...")
        
        # Create test batches for different manufacturers
        test_batches = [
            {
                "batch_id": "BATCH_MFG1_001",
                "product_name": "Organic Apples",
                "quantity": 100.0,
                "harvest_date": datetime.now() - timedelta(days=3),
                "origin": "Farm A",
                "manufacturer_id": "manufacturer1",
                "current_location": "Warehouse A",
                "status": "In Transit"
            },
            {
                "batch_id": "BATCH_MFG1_002", 
                "product_name": "Fresh Bananas",
                "quantity": 150.0,
                "harvest_date": datetime.now() - timedelta(days=2),
                "origin": "Farm B",
                "manufacturer_id": "manufacturer1",
                "current_location": "Distribution Center",
                "status": "Delivered"
            },
            {
                "batch_id": "BATCH_MFG2_001",
                "product_name": "Organic Carrots",
                "quantity": 200.0,
                "harvest_date": datetime.now() - timedelta(days=1),
                "origin": "Farm C", 
                "manufacturer_id": "manufacturer2",
                "current_location": "Factory",
                "status": "Created"
            }
        ]
        
        # Insert test batches
        for batch in test_batches:
            await conn.execute("""
                INSERT INTO batches 
                (batch_id, product_name, quantity, harvest_date, origin, manufacturer_id, current_location, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (batch_id) DO UPDATE SET
                product_name = EXCLUDED.product_name,
                manufacturer_id = EXCLUDED.manufacturer_id,
                current_location = EXCLUDED.current_location,
                status = EXCLUDED.status
            """, batch["batch_id"], batch["product_name"], batch["quantity"], 
                batch["harvest_date"], batch["origin"], batch["manufacturer_id"],
                batch["current_location"], batch["status"])
        
        print("✓ Created test batches")
        
        # Create test scan records for transporters and retailers
        test_scans = [
            # Transporter1 scans from manufacturer1
            {
                "scan_id": str(uuid.uuid4()),
                "batch_id": "BATCH_MFG1_001",
                "operator_id": "transporter1",
                "operator_type": "transporter",
                "scan_type": "pickup",
                "vehicle_id": "TRUCK_001",
                "location": "Warehouse A",
                "timestamp": datetime.now() - timedelta(hours=6)
            },
            {
                "scan_id": str(uuid.uuid4()),
                "batch_id": "BATCH_MFG1_002",
                "operator_id": "transporter1", 
                "operator_type": "transporter",
                "scan_type": "pickup",
                "vehicle_id": "TRUCK_001",
                "location": "Warehouse A",
                "timestamp": datetime.now() - timedelta(hours=4)
            },
            
            # Transporter2 scans different batch
            {
                "scan_id": str(uuid.uuid4()),
                "batch_id": "BATCH_MFG2_001",
                "operator_id": "transporter2",
                "operator_type": "transporter", 
                "scan_type": "pickup",
                "vehicle_id": "TRUCK_002",
                "location": "Factory",
                "timestamp": datetime.now() - timedelta(hours=3)
            },
            
            # Retailer1 receives from transporter1
            {
                "scan_id": str(uuid.uuid4()),
                "batch_id": "BATCH_MFG1_001",
                "operator_id": "retailer1",
                "operator_type": "retailer",
                "scan_type": "receive",
                "location": "Store A",
                "timestamp": datetime.now() - timedelta(hours=2)
            },
            {
                "scan_id": str(uuid.uuid4()),
                "batch_id": "BATCH_MFG1_002",
                "operator_id": "retailer1",
                "operator_type": "retailer", 
                "scan_type": "receive",
                "location": "Store A",
                "timestamp": datetime.now() - timedelta(hours=1)
            },
            
            # Retailer2 receives different batch
            {
                "scan_id": str(uuid.uuid4()),
                "batch_id": "BATCH_MFG2_001",
                "operator_id": "retailer2",
                "operator_type": "retailer",
                "scan_type": "receive", 
                "location": "Store B",
                "timestamp": datetime.now() - timedelta(minutes=30)
            }
        ]
        
        # Insert test scan records
        for scan in test_scans:
            await conn.execute("""
                INSERT INTO qr_scan_records
                (scan_id, batch_id, operator_id, operator_type, scan_type, vehicle_id, location, timestamp)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (scan_id) DO UPDATE SET
                operator_id = EXCLUDED.operator_id,
                operator_type = EXCLUDED.operator_type
            """, scan["scan_id"], scan["batch_id"], scan["operator_id"], 
                scan["operator_type"], scan["scan_type"], scan.get("vehicle_id"),
                scan["location"], scan["timestamp"])
        
        print("✓ Created test scan records")
        
        # Print summary of what each user should see
        print("\n📊 Expected role-based access:")
        print("👨‍🏭 manufacturer1: Should see BATCH_MFG1_001, BATCH_MFG1_002")
        print("👨‍🏭 manufacturer2: Should see BATCH_MFG2_001") 
        print("🚛 transporter1: Should see BATCH_MFG1_001, BATCH_MFG1_002")
        print("🚛 transporter2: Should see BATCH_MFG2_001")
        print("🏪 retailer1: Should see BATCH_MFG1_001, BATCH_MFG1_002") 
        print("🏪 retailer2: Should see BATCH_MFG2_001")
        
        # Test queries to verify filtering works
        print("\n🔍 Verification queries:")
        
        # Manufacturer1 batches
        manufacturer1_batches = await conn.fetch("""
            SELECT batch_id, product_name FROM batches 
            WHERE manufacturer_id = 'manufacturer1'
        """)
        print(f"Manufacturer1 batches: {[b['batch_id'] for b in manufacturer1_batches]}")
        
        # Transporter1 batches
        transporter1_batches = await conn.fetch("""
            SELECT DISTINCT b.batch_id, b.product_name FROM batches b
            INNER JOIN qr_scan_records qsr ON b.batch_id = qsr.batch_id
            WHERE qsr.operator_id = 'transporter1' AND qsr.operator_type = 'transporter'
        """)
        print(f"Transporter1 batches: {[b['batch_id'] for b in transporter1_batches]}")
        
        # Retailer1 batches
        retailer1_batches = await conn.fetch("""
            SELECT DISTINCT b.batch_id, b.product_name FROM batches b
            INNER JOIN qr_scan_records qsr ON b.batch_id = qsr.batch_id
            WHERE qsr.operator_id = 'retailer1' AND qsr.operator_type = 'retailer'
        """)
        print(f"Retailer1 batches: {[b['batch_id'] for b in retailer1_batches]}")
        
        print("\n✅ Test data creation completed!")
        
    except Exception as e:
        print(f"❌ Error creating test data: {e}")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(create_test_data())