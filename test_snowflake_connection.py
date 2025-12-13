#!/usr/bin/env python3
"""Test Snowflake connection with detailed diagnostics"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import snowflake.connector

load_dotenv()

# Get config
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")
SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE")

print("=" * 60)
print("Snowflake Connection Diagnostic Test")
print("=" * 60)
print(f"\nConfiguration:")
print(f"  Account: {SNOWFLAKE_ACCOUNT}")
print(f"  User: {SNOWFLAKE_USER}")
print(f"  Database: {SNOWFLAKE_DATABASE}")
print(f"  Schema: {SNOWFLAKE_SCHEMA}")
print(f"  Warehouse: {SNOWFLAKE_WAREHOUSE}")
print(f"  Role: {SNOWFLAKE_ROLE}")

# Get current IP
try:
    import requests
    current_ip = requests.get('https://api.ipify.org', timeout=5).text
    print(f"\n  Your IP: {current_ip}")
except:
    current_ip = "Unknown"

print("\n" + "=" * 60)
print("Test 1: Direct Snowflake Connector")
print("=" * 60)

# Try direct connector
try:
    conn = snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT.strip(),
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA,
        role=SNOWFLAKE_ROLE,
        login_timeout=20,
        network_timeout=20
    )
    cursor = conn.cursor()
    cursor.execute("SELECT CURRENT_ACCOUNT(), CURRENT_USER(), CURRENT_ROLE()")
    result = cursor.fetchone()
    print(f"✅ SUCCESS with direct connector!")
    print(f"   Account: {result[0]}")
    print(f"   User: {result[1]}")
    print(f"   Role: {result[2]}")
    cursor.close()
    conn.close()
except Exception as e:
    print(f"❌ FAILED: {str(e)}")
    print(f"\n⚠️  This indicates a network connectivity issue.")
    print(f"   Your IP ({current_ip}) may not be whitelisted, or there's a firewall blocking the connection.")
    print(f"\n   Verify in Snowflake:")
    print(f"   ```sql")
    print(f"   SHOW USERS LIKE '{SNOWFLAKE_USER}';")
    print(f"   SELECT \"name\", \"network_policy\" FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()));")
    print(f"   ```")

print("\n" + "=" * 60)
print("Test 2: SQLAlchemy Connection")
print("=" * 60)

# Try SQLAlchemy
try:
    password_encoded = quote_plus(SNOWFLAKE_PASSWORD)
    connection_string = f"snowflake://{SNOWFLAKE_USER}:{password_encoded}@{SNOWFLAKE_ACCOUNT.strip()}/{SNOWFLAKE_DATABASE}/{SNOWFLAKE_SCHEMA}?warehouse={SNOWFLAKE_WAREHOUSE}"
    if SNOWFLAKE_ROLE:
        connection_string += f"&role={SNOWFLAKE_ROLE}"
    
    engine = create_engine(connection_string, pool_pre_ping=True)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT CURRENT_ACCOUNT(), CURRENT_USER()"))
        info = result.fetchone()
        print(f"✅ SUCCESS with SQLAlchemy!")
        print(f"   Account: {info[0]}")
        print(f"   User: {info[1]}")
except Exception as e:
    print(f"❌ FAILED: {str(e)}")

print("\n" + "=" * 60)
print("Diagnostic Complete")
print("=" * 60)

