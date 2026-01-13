import psycopg2

try:
    conn = psycopg2.connect('postgresql://postgres:123@localhost:5432/traceabilitydb')
    cur = conn.cursor()
    
    cur.execute('SELECT username, password_hash FROM users LIMIT 5;')
    print('Users and their password hashes:')
    for row in cur.fetchall():
        print(f'Username: {row[0]}, Hash: {row[1][:50]}...')
    
    conn.close()
    print('\nDatabase connection successful')
except Exception as e:
    print(f'Error: {e}')