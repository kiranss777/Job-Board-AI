-- Snowflake IP Whitelisting SQL Commands
-- Run these commands as ACCOUNTADMIN in Snowflake Web UI

-- Step 1: Check current network policies
SHOW NETWORK POLICIES;

-- Step 2: Check if your user has a network policy assigned
SHOW USERS LIKE 'Vemana30';
SELECT "name", "network_policy" FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()));

-- Step 3: Create or replace a network policy to allow your IP
-- Replace 'YOUR_IP_HERE' with your actual IP address (check UI for current IP)
CREATE OR REPLACE NETWORK POLICY ALLOW_CURRENT_IP
  ALLOWED_IP_LIST = ('0.0.0.0/0')  -- Allows all IPs (for testing - change to your specific IP for production)
  BLOCKED_IP_LIST = ();

-- Step 4: Apply the network policy to your user
ALTER USER Vemana30 SET NETWORK_POLICY = ALLOW_CURRENT_IP;

-- Step 5: Verify the policy is applied
SHOW USERS LIKE 'Vemana30';
SELECT "name", "network_policy" FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()));

-- Alternative: If you want to allow only your specific IP (more secure)
-- Replace '73.186.187.112' with your actual IP from the UI
-- CREATE OR REPLACE NETWORK POLICY ALLOW_CURRENT_IP
--   ALLOWED_IP_LIST = ('73.186.187.112/32')
--   BLOCKED_IP_LIST = ();

