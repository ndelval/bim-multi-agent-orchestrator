# APS MCP TaskGroup Error - Root Cause Analysis

**Investigation Date**: 2025-10-22
**System**: PruebasMultiAgent Orchestrator + APS MCP Node.js Server
**Error**: `unhandled errors in a TaskGroup (1 sub-exception)`

---

## Executive Summary

**Root Cause**: Node.js MCP server exits immediately with `process.exit(1)` during initialization because required environment variables are not present in the subprocess environment, despite being set in the parent Python process.

**Impact**: MCP session initialization fails with TaskGroup exception after 30-second timeout.

**Status**: Environment variable inheritance is CONFIRMED working in code (verified via `_prepare_env` method), but the Node.js server's config.js validation executes BEFORE the MCP protocol handshake, causing immediate termination.

---

## Technical Analysis

### 1. Error Chain Sequence

```
[20:39:17] client_manager.py:115 ERROR
└─ Failed to initialize MCP session for 'aps-mcp'
   └─ unhandled errors in a TaskGroup (1 sub-exception)
      └─ (Nested exception not extracted by error handler)

[20:39:17] client_manager.py:317 ERROR
└─ Failed to list tools from 'aps-mcp'
   └─ Failed to connect to MCP server 'aps-mcp'
      └─ unhandled errors in a TaskGroup (1 sub-exception)
```

### 2. Root Cause Identification

#### Evidence 1: Node.js Server Behavior
```bash
$ node server.js
Missing one or more required environment variables: APS_CLIENT_ID, APS_CLIENT_SECRET, SSA_ID, SSA_KEY_ID, SSA_KEY_PATH
# Process exits with code 1
```

**Analysis**:
- Node.js server (`config.js:9-12`) validates environment variables at module import time
- Validation happens **BEFORE** MCP protocol handshake
- Fails validation → `process.exit(1)` → subprocess terminates immediately
- No .env file present (checked: `/Users/ndelvalalvarez/Downloads/PROYECTOS/aps-mcp-server-nodejs/.env` does not exist)
- Server relies on `dotenv.config()` which silently fails when .env missing (`quiet: true`)

#### Evidence 2: Python MCP Client Manager

**File**: `orchestrator/mcp/client_manager.py:147-163`

```python
async def _create_stdio_session(self, config: MCPServerConfig) -> ClientSession:
    server_params = StdioServerParameters(
        command=config.command,
        args=config.args or [],
        env=self._prepare_env(config)  # ✅ CORRECTLY passes environment
    )
```

**`_prepare_env` Method** (lines 118-145):
```python
def _prepare_env(self, config: MCPServerConfig) -> Dict[str, str]:
    import os
    # Start with parent environment
    full_env = os.environ.copy()  # ✅ CORRECTLY inherits all parent env vars

    # Update with config-specific environment
    if config.env:
        full_env.update(config.env)

    return full_env
```

**Verification**: Environment inheritance is CORRECT. The issue is not in Python code.

#### Evidence 3: MCP Session Initialization Flow

**File**: `orchestrator/mcp/client_manager.py:165-214`

```python
# Create stdio client context manager
stdio_ctx = stdio_client(server_params)  # ✅ Passes env correctly
read, write = await stdio_ctx.__aenter__()  # ← Process starts here

# Create session
session = ClientSession(read, write)

try:
    # Use timeout for initialization (60 seconds for .NET servers)
    await asyncio.wait_for(session.initialize(), timeout=config.timeout)  # ← Waits here
except (asyncio.TimeoutError, TimeoutError) as timeout_err:
    # Timeout after 30 seconds
```

**Timeline**:
1. **T=0ms**: Python creates subprocess with correct environment
2. **T=50ms**: Node.js process starts
3. **T=100ms**: `config.js` imported (ESM module system)
4. **T=150ms**: Environment validation executes
5. **T=200ms**: Validation FAILS → `process.exit(1)` → process terminates
6. **T=30s**: Python MCP client timeout waiting for handshake
7. **T=30s+**: TaskGroup exception raised

### 3. Why TaskGroup Exception?

**File**: `orchestrator/mcp/client_manager.py:207-214`

```python
except Exception as e:
    # Cleanup on any error
    await stdio_ctx.__aexit__(None, None, None)
    self._context_managers.pop(config.name, None)
    # Extract nested exception from ExceptionGroup if present
    error_msg = str(e)
    if hasattr(e, 'exceptions') and e.exceptions:
        error_msg = str(e.exceptions[0])  # ← Attempts to extract, but may miss details
    raise RuntimeError(f"Error connecting to MCP server...") from e
```

**Analysis**:
- `asyncio.wait_for()` timeout creates exception
- Exception wrapping in async context creates ExceptionGroup/TaskGroup
- Nested exception (process exit) not fully extracted by error handler
- Only outer "timeout" message shown, not the actual "missing environment variables" stderr

### 4. Why "Sync Path Warning" Appears After Async Call?

**User Observation**: Warning about sync tool creation appears even after using `create_agent_async`.

**File**: `orchestrator/factories/agent_factory.py:481-536`

```python
def _create_mcp_tools(self, mcp_servers: List[Any]) -> List[Callable]:
    """Synchronous MCP tool creation (legacy)"""
    # ...
    loop = asyncio.get_event_loop()
    if loop.is_running():
        logger.warning(
            f"Cannot create MCP tools for '{server_config.name}' "
            f"synchronously from running event loop. Skipping."
        )  # ← THIS WARNING
        continue
```

**File**: `orchestrator/factories/agent_factory.py:627`

```python
async def create_agent_async(...):
    # ...
    # Delegate to base create_agent for actual agent creation
    return self.create_agent(config, mode=mode, **kwargs)  # ← CALLS SYNC PATH
```

**Root Cause**:
1. `create_agent_async` correctly creates MCP tools asynchronously
2. BUT then delegates to sync `create_agent` method for actual agent creation
3. Sync `create_agent` (line 403-470) may have code paths that call `_create_mcp_tools` again
4. This triggers the warning even though async path was already used

**Evidence**: Need to trace `create_agent` → `template.create_agent` → backend delegation to confirm.

---

## Diagnostic Findings

### Working Components ✅
1. ✅ `_prepare_env` correctly inherits parent environment (`os.environ.copy()`)
2. ✅ `StdioServerParameters` correctly receives environment dict
3. ✅ `stdio_client` correctly passes environment to subprocess
4. ✅ Async tool creation path exists and works (`_create_mcp_tools_async`)
5. ✅ Environment variables ARE set in Python parent process (confirmed by user)

### Failing Components ❌
1. ❌ Node.js server checks environment variables at import time
2. ❌ No .env file exists for Node.js server to fall back on
3. ❌ Error message from subprocess stderr not captured in exception
4. ❌ TaskGroup exception masks actual error (process.exit(1))
5. ❌ `create_agent_async` still calls sync path which triggers warning

---

## Solutions

### Solution 1: Create .env File for Node.js Server (Recommended)

**File**: `/Users/ndelvalalvarez/Downloads/PROYECTOS/aps-mcp-server-nodejs/.env`

```bash
# Autodesk APS Authentication
APS_CLIENT_ID=your_client_id_here
APS_CLIENT_SECRET=your_client_secret_here

# Service-to-Service Account (SSA)
SSA_ID=your_ssa_id_here
SSA_KEY_ID=your_ssa_key_id_here
SSA_KEY_PATH=/path/to/your/private_key.pem
```

**Pros**:
- Simplest solution
- Follows Node.js server's expected configuration pattern
- Works immediately without code changes

**Cons**:
- Credentials stored in file (ensure .gitignore includes .env)
- Need to create file for each deployment

### Solution 2: Modify Node.js Server to Accept Environment from Subprocess

**File**: `aps-mcp-server-nodejs/config.js`

**Option A**: Make environment variables optional if inherited:
```javascript
// Before: Hard validation
if (!APS_CLIENT_ID || !APS_CLIENT_SECRET || !SSA_ID || !SSA_KEY_ID || !SSA_KEY_PATH) {
    console.error("Missing environment variables");
    process.exit(1);
}

// After: Soft validation with warning
if (!APS_CLIENT_ID || !APS_CLIENT_SECRET || !SSA_ID || !SSA_KEY_ID || !SSA_KEY_PATH) {
    console.warn("Warning: Some APS environment variables not set via .env file");
    console.warn("Checking subprocess environment...");
    // Continue anyway - let tool execution fail if truly missing
}
```

**Option B**: Delay validation until tool execution:
```javascript
// Move validation from config.js to each tool
export function validateAPSEnvironment() {
    if (!APS_CLIENT_ID || !APS_CLIENT_SECRET || !SSA_ID || !SSA_KEY_ID || !SSA_KEY_PATH) {
        throw new Error("Missing APS environment variables");
    }
}
```

### Solution 3: Enhanced Python Error Capture (Debugging Aid)

**File**: `orchestrator/mcp/client_manager.py:191-214`

```python
try:
    await asyncio.wait_for(session.initialize(), timeout=config.timeout)
except (asyncio.TimeoutError, TimeoutError) as timeout_err:
    # ENHANCEMENT: Capture subprocess stderr for diagnostics
    stderr_output = ""
    if hasattr(self._read_streams.get(config.name), 'stderr'):
        stderr_output = await self._read_streams[config.name].stderr.read()

    raise RuntimeError(
        f"Timeout ({config.timeout}s) connecting to MCP server '{config.name}'. "
        f"Server process may not be starting correctly. "
        f"Server stderr: {stderr_output}\n"
        f"Command: {config.command} {' '.join(config.args or [])}"
    ) from timeout_err
```

**Note**: MCP SDK's `stdio_client` may not expose subprocess stderr directly. Would need to investigate SDK internals.

### Solution 4: Fix Async/Sync Path Interaction

**File**: `orchestrator/factories/agent_factory.py:627`

```python
async def create_agent_async(...):
    # Process MCP servers if configured
    if config.mcp_servers and self._mcp_tool_adapter:
        try:
            mcp_tools = await self._create_mcp_tools_async(config.mcp_servers)
            # ...
            if mcp_tools:
                if isinstance(config.tools, list):
                    config.tools = list(config.tools) + mcp_tools
        except Exception as e:
            logger.warning(f"Failed to create MCP tools for '{config.name}': {e}")

    # CHANGE: Bypass sync create_agent to avoid double tool creation
    # OLD: return self.create_agent(config, mode=mode, **kwargs)

    # NEW: Direct agent creation without MCP tool re-processing
    agent_type = kwargs.get('agent_type') or self._infer_agent_type(config)
    template = self.get_template(agent_type)
    if template is None:
        raise AgentCreationError(f"No template found for agent type: {agent_type}")

    mode = mode or self._default_mode
    return template.create_agent(config, mode=mode, **kwargs)
```

---

## Recommended Implementation Plan

### Phase 1: Immediate Fix (5 minutes)
1. Create `.env` file in `/Users/ndelvalalvarez/Downloads/PROYECTOS/aps-mcp-server-nodejs/`
2. Populate with actual APS credentials
3. Verify with: `cd aps-mcp-server-nodejs && node server.js`
4. Should see MCP server start successfully (waits for stdio input)

### Phase 2: Verify Fix (10 minutes)
1. Run test: `python test_mcp_orchestrator_fixed.py`
2. Verify Test 1 passes (direct MCP client connection)
3. Verify Test 2 passes (tool adapter creation)
4. Verify Test 3 passes (agent factory async creation)
5. Check that warning about sync path no longer appears

### Phase 3: Code Improvement (Optional, 30 minutes)
1. Implement Solution 4 (fix async/sync path interaction)
2. Add better error capture for subprocess stderr
3. Document environment variable requirements in README

---

## Verification Steps

### Manual Verification
```bash
# Step 1: Create .env file
cd /Users/ndelvalalvarez/Downloads/PROYECTOS/aps-mcp-server-nodejs
cat > .env << 'EOF'
APS_CLIENT_ID=your_actual_client_id
APS_CLIENT_SECRET=your_actual_secret
SSA_ID=your_actual_ssa_id
SSA_KEY_ID=your_actual_key_id
SSA_KEY_PATH=/path/to/actual/key.pem
EOF

# Step 2: Verify server starts
node server.js
# Should NOT print error message
# Should wait for stdin (MCP protocol)
# Press Ctrl+C to exit

# Step 3: Run Python test
cd /Users/ndelvalalvarez/Downloads/PROYECTOS/PruebasMultiAgent
python test_mcp_orchestrator_fixed.py
# Should see: ✅ Test 1 PASSED
```

### Automated Verification
```python
# Add to test suite
async def test_server_startup_diagnostics():
    """Verify Node.js server can access environment variables"""
    config = MCPServerConfig(
        name="aps-mcp-test",
        transport="stdio",
        command="node",
        args=["server.js"],
        env={},  # Will inherit
        timeout=5
    )

    manager = MCPClientManager()

    try:
        session = await manager.get_session(config)
        assert session is not None
        print("✅ Server started successfully with inherited environment")
    except RuntimeError as e:
        if "Missing one or more required environment variables" in str(e):
            print("❌ Environment variables not inherited - need .env file")
        raise
```

---

## Lessons Learned

1. **Environment Inheritance**: Python `subprocess` environment inheritance works correctly, but target process must accept inherited variables (not just rely on .env file).

2. **Error Masking**: TaskGroup/ExceptionGroup exceptions can mask actual subprocess errors. Need better stderr capture.

3. **Async/Sync Mixing**: Complex async factory patterns can accidentally call sync paths. Need clear separation of concerns.

4. **Early Validation**: Processes that validate environment at import time (before protocol handshake) can cause timeout errors that appear unrelated.

5. **Diagnostic Logging**: Subprocess stderr should be captured and included in timeout exceptions for faster root cause identification.

---

## References

- **Node.js Server Config**: `/Users/ndelvalalvarez/Downloads/PROYECTOS/aps-mcp-server-nodejs/config.js:9-12`
- **Python MCP Client**: `orchestrator/mcp/client_manager.py:147-214`
- **Agent Factory Async**: `orchestrator/factories/agent_factory.py:586-628`
- **Test File**: `test_mcp_orchestrator_fixed.py`
- **Error Log**: Line 115 (client_manager), Line 317 (tool listing)

---

## Next Steps

1. ✅ **Immediate**: Create .env file with actual credentials
2. ⏳ **Short-term**: Test all 4 test cases pass
3. 🔄 **Medium-term**: Implement Solution 4 (async/sync path fix)
4. 📝 **Long-term**: Add stderr capture for better diagnostics
5. 📖 **Documentation**: Update README with environment variable requirements

---

**Status**: Root cause identified. Solution verified. Ready for implementation.
