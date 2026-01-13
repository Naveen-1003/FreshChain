import asyncpg
import asyncio

async def test_db():
    try:
        conn = await asyncpg.connect(
            host="localhost",
            port=5432,
            user="postgres",
            password="password",
            database="visaion"
        )
        print("✅ Database connection successful")
        
        # List tables
        tables = await conn.fetch("SELECT tablename FROM pg_tables WHERE schemaname='public';")
        print(f"📋 Found {len(tables)} tables:", [t[0] for t in tables])
        
        # Check batches table
        try:
            batches = await conn.fetch("SELECT COUNT(*) FROM batches;")
            print(f"📦 Batches table has {batches[0][0]} records")
        except Exception as e:
            print(f"❌ Batches table issue: {e}")
        
        # Check if scanned_by_retailer column exists
        try:
            columns = await conn.fetch("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'batches' AND table_schema = 'public';
            """)
            column_names = [c[0] for c in columns]
            print(f"🔍 Batches table columns: {column_names}")
            if 'scanned_by_retailer' in column_names:
                print("✅ scanned_by_retailer column exists")
            else:
                print("❌ scanned_by_retailer column missing")
        except Exception as e:
            print(f"❌ Column check failed: {e}")
        
        await conn.close()
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_db())