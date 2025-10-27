# APS MCP Integration - Quick Fix Guide

**Problem**: MCP server fails with TaskGroup exception
**Root Cause**: Node.js server exits immediately due to missing environment variables
**Solution**: Create .env file for Node.js server
**Time**: 5 minutes

---

## Step 1: Verify Environment Variables

Check that you have the required APS credentials:

```bash
# In your shell, verify these are set:
echo $APS_CLIENT_ID
echo $APS_CLIENT_SECRET
echo $SSA_ID
echo $SSA_KEY_ID
echo $SSA_KEY_PATH
```

If any are empty, you need to obtain these credentials from Autodesk Platform Services.

---

## Step 2: Create .env File

Create the .env file in the Node.js server directory:

```bash
cd /Users/ndelvalalvarez/Downloads/PROYECTOS/aps-mcp-server-nodejs

cat > .env << 'EOF'
# Autodesk Platform Services Authentication
APS_CLIENT_ID=<your_client_id>
APS_CLIENT_SECRET=<your_client_secret>

# Service-to-Service Account (SSA)
SSA_ID=<your_ssa_id>
SSA_KEY_ID=<your_ssa_key_id>
SSA_KEY_PATH=<path_to_your_private_key.pem>
EOF
```

**Replace the placeholders** with your actual values from Step 1.

---

## Step 3: Verify Server Starts

Test that the Node.js server can now start:

```bash
cd /Users/ndelvalalvarez/Downloads/PROYECTOS/aps-mcp-server-nodejs
node server.js
```

**Expected Output**:
- No error messages
- Process waits for input (stdio server mode)
- Press `Ctrl+C` to exit

**If you see**: `Missing one or more required environment variables...`
→ Your .env file has incorrect values or paths. Check Step 2 again.

---

## Step 4: Run Python Tests

Now test the MCP integration from Python:

```bash
cd /Users/ndelvalalvarez/Downloads/PROYECTOS/PruebasMultiAgent
python test_mcp_orchestrator_fixed.py
```

**Expected Output**:
```
╔══════════════════════════════════════════════════════════╗
║          MCP Orchestrator Fix Verification               ║
╚══════════════════════════════════════════════════════════╝

============================================================
Test 1: Direct MCP Client Connection
============================================================

🔗 Connecting to APS MCP server...
✅ Connected successfully

📋 Listing tools...
Found X tools:
  - getProjectsTool
  - ...

🔧 Calling getProjectsTool...
Result type: <class 'mcp.types.CallToolResult'>
...

✅ Test 1 PASSED
```

If **Test 1 fails**, check the error message for details.

---

## Step 5: Verify No More Warnings

The warning about sync tool creation should no longer appear:

```bash
# Previous behavior (BEFORE fix):
WARNING: Cannot create MCP tools synchronously from running event loop

# Expected behavior (AFTER fix):
✅ Created N MCP tool(s) for agent 'AgentName'
```

---

## Troubleshooting

### Issue: "Module not found: dotenv"

**Solution**: Install dependencies in the test directory:
```bash
pip install python-dotenv
```

### Issue: "Permission denied" for SSA_KEY_PATH

**Solution**: Verify the PEM file exists and is readable:
```bash
ls -l /path/to/your/private_key.pem
chmod 600 /path/to/your/private_key.pem
```

### Issue: Server connects but tools fail

**Solution**: Check APS credentials are valid:
- Log into Autodesk Platform Services portal
- Verify Client ID and Secret
- Verify SSA configuration
- Ensure private key matches the SSA_KEY_ID

### Issue: Still getting TaskGroup exception

**Solution**: Enable verbose logging to see actual error:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Then check the logs for the actual subprocess error message.

---

## Why This Works

**Problem Details**:
1. Node.js MCP server has `config.js` that validates environment variables at import time
2. Python MCP client correctly inherits environment to subprocess
3. BUT Node.js server relies on `dotenv` package to load `.env` file
4. When `.env` file is missing, `dotenv.config({quiet: true})` silently fails
5. Environment validation then checks `process.env` (which should have inherited vars)
6. However, Node.js ESM module import happens BEFORE subprocess environment is fully available
7. Result: Validation fails → `process.exit(1)` → Python client timeout → TaskGroup exception

**Solution**:
- Creating `.env` file ensures Node.js server has credentials from file-based source
- This bypasses the environment inheritance timing issue
- Server starts successfully and can participate in MCP protocol handshake

---

## Security Note

⚠️ **Important**: The `.env` file contains sensitive credentials.

**Ensure**:
1. `.env` is listed in `.gitignore`
2. File permissions are restrictive: `chmod 600 .env`
3. Do not commit `.env` to version control
4. Use environment variables in production (not .env files)

Check your `.gitignore`:
```bash
cd /Users/ndelvalalvarez/Downloads/PROYECTOS/aps-mcp-server-nodejs
grep -q "^\.env$" .gitignore || echo ".env" >> .gitignore
```

---

## Alternative: Environment-Only Approach

If you prefer NOT to use .env files, you need to modify the Node.js server:

**File**: `aps-mcp-server-nodejs/config.js`

```javascript
// BEFORE (strict validation):
if (!APS_CLIENT_ID || !APS_CLIENT_SECRET || !SSA_ID || !SSA_KEY_ID || !SSA_KEY_PATH) {
    console.error("Missing one or more required environment variables...");
    process.exit(1);
}

// AFTER (lenient validation):
if (!APS_CLIENT_ID || !APS_CLIENT_SECRET || !SSA_ID || !SSA_KEY_ID || !SSA_KEY_PATH) {
    console.warn("Warning: Some environment variables not set via .env");
    console.warn("Will check subprocess environment at runtime");
    // Continue - let tool execution fail if truly missing
}
```

Then ensure Python passes environment explicitly:
```python
mcp_config = MCPServerConfig(
    name="aps-mcp",
    command="node",
    args=["server.js"],
    env={  # Explicit environment (not empty dict)
        "APS_CLIENT_ID": os.getenv("APS_CLIENT_ID"),
        "APS_CLIENT_SECRET": os.getenv("APS_CLIENT_SECRET"),
        "SSA_ID": os.getenv("SSA_ID"),
        "SSA_KEY_ID": os.getenv("SSA_KEY_ID"),
        "SSA_KEY_PATH": os.getenv("SSA_KEY_PATH"),
    }
)
```

---

## Verification Checklist

- [ ] Step 1: Environment variables verified in shell
- [ ] Step 2: .env file created with correct values
- [ ] Step 3: Node.js server starts without errors
- [ ] Step 4: Python Test 1 passes (direct connection)
- [ ] Step 4: Python Test 2 passes (tool adapter)
- [ ] Step 4: Python Test 3 passes (agent factory)
- [ ] Step 5: No sync tool creation warnings
- [ ] Security: .env file is in .gitignore
- [ ] Security: .env file has chmod 600 permissions

---

**Status**: Ready to implement. Follow steps 1-5 in sequence.
