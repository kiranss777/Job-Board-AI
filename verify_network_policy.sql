-- Run these SQL commands in Snowflake Web UI to verify network policy is applied

-- Step 1: Check all network policies
SHOW NETWORK POLICIES;

-- Step 2: Check your user's network policy
SHOW USERS LIKE 'Vemana30';
SELECT "name", "network_policy" FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()));

-- Step 3: If no policy is assigned, create and apply one
-- Option A: Allow all IPs (for testing)
CREATE OR REPLACE NETWORK POLICY ALLOW_ALL_IPS
  ALLOWED_IP_LIST = ('0.0.0.0/0')
  BLOCKED_IP_LIST = ();

ALTER USER Vemana30 SET NETWORK_POLICY = ALLOW_ALL_IPS;

-- Option B: Allow only your specific IP (more secure)
-- Replace '73.186.187.112' with your actual IP
CREATE OR REPLACE NETWORK POLICY ALLOW_CURRENT_IP
  ALLOWED_IP_LIST = ('73.186.187.112/32')
  BLOCKED_IP_LIST = ();

ALTER USER Vemana30 SET NETWORK_POLICY = ALLOW_CURRENT_IP;

-- Step 4: Verify the policy is now applied
SHOW USERS LIKE 'Vemana30';
SELECT "name", "network_policy" FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()));

-- Step 5: Check if there are any account-level network policies that might override
SHOW PARAMETERS LIKE 'NETWORK_POLICY' IN ACCOUNT;

