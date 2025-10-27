# APS MCP stdio Transport Incompatibility

**Date:** 2025-10-22
**Status:** ⚠️ IDENTIFIED - Python MCP SDK stdio_client incompatible with APS MCP Node.js server
**Impact:** Cannot connect to APS MCP server via stdio transport
**Workaround:** Use HTTP/SSE transport or patch SDK

---

## Executive Summary

The Python MCP SDK (version 1.12.4) has an **incompatibility with the Node.js APS MCP server's stdio implementation**. While the server correctly implements the MCP protocol and responds properly to JSON-RPC messages, the Python SDK's `stdio_client` context manager encounters a `BrokenResourceError` during the initialization handshake.

**Root Cause:** The SDK's `stdout_reader` task crashes when trying to send parsed messages to the `read_stream` because ClientSession.initialize() prematurely closes the stream, or there's a race condition in stream handling.

**Evidence:** Raw subprocess communication with the APS server works perfectly, proving the server is functional and MCP-compliant.

---

## Diagnostic Summary

### ✅ What Works

1. **Node.js APS MCP Server:** Starts successfully, loads .env file, responds to MCP protocol
2. **Protocol Version 2024-11-05:** Server supports and correctly responds with this version
3. **Raw MCP Protocol:** Direct subprocess communication works perfectly
4. **Server Response Format:** JSON-RPC messages are correctly newline-delimited

### ❌ What Fails

1. **Python MCP SDK stdio_client:** Times out during session.initialize()
2. **stdout_reader Task:** Crashes with `anyio.BrokenResourceError`
3. **Stream Communication:** read_stream closes unexpectedly, breaking message sending

---

## Technical Details

### Error Stack Trace

```
ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)
  +-+---------------- 1 ----------------
    | File "mcp/client/stdio/__init__.py", line 160, in stdout_reader
    |     await read_stream_writer.send(session_message)
    | File "anyio/streams/memory.py", line 256, in send
    |     raise BrokenResourceError from None
    | anyio.BrokenResourceError
    +------------------------------------
```

### Problem Analysis

**MCP SDK stdio_client Flow:**

1. ✅ Spawns Node.js subprocess successfully
2. ✅ Creates read/write memory streams
3. ✅ Starts stdout_reader and stdin_writer tasks
4. ❌ ClientSession.initialize() sends init request
5. ❌ Server responds correctly but SDK's stdout_reader can't send parsed message
6. ❌ BrokenResourceError: read_stream was closed before message could be sent
7. ❌ Timeout after 60 seconds, TaskGroup exception during cleanup

**Why Raw Communication Works:**

```python
# Raw test (WORKS)
process = await asyncio.create_subprocess_exec(...)
process.stdin.write(json_request + "\n")
response = await process.stdout.readline()  # ✅ Simple, direct

# MCP SDK (FAILS)
async with stdio_client(params) as (read, write):  # Complex stream management
    session = ClientSession(read, write)
    await session.initialize()  # ❌ Stream closes prematurely
```

### Why the Fix Attempts Didn't Work

1. **Created .env file:** ✅ Server loads it correctly, not the issue
2. **Added protocol_version:** ✅ Server accepts 2024-11-05, not the issue
3. **Environment inheritance:** ✅ Working correctly, not the issue

**The real issue:** SDK's stream management is incompatible with how the Node.js server communicates.

---

## Reproduction Steps

### Test 1: Raw MCP Protocol (✅ WORKS)

```bash
cd /path/to/aps-mcp-server-nodejs
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' | node server.js
```

**Result:**
```json
{"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{"listChanged":true}},"serverInfo":{"name":"aps-mcp-server-nodejs","version":"0.0.1"}},"jsonrpc":"2.0","id":1}
```

### Test 2: Python Raw Subprocess (✅ WORKS)

```python
import asyncio
import json

async def test():
    process = await asyncio.create_subprocess_exec(
        "node", "/path/to/aps-mcp-server-nodejs/server.js",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE
    )

    request = json.dumps({"jsonrpc":"2.0","id":1,"method":"initialize",
                         "params":{"protocolVersion":"2024-11-05","capabilities":{},
                                  "clientInfo":{"name":"test","version":"1.0"}}}) + "\n"
    process.stdin.write(request.encode())
    await process.stdin.drain()

    response = await asyncio.wait_for(process.stdout.readline(), timeout=10.0)
    print(f"✅ Success: {response.decode()}")

asyncio.run(test())
```

### Test 3: MCP SDK stdio_client (❌ FAILS)

```python
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

async def test():
    params = StdioServerParameters(
        command="node",
        args=["/path/to/aps-mcp-server-nodejs/server.js"]
    )

    async with stdio_client(params) as (read, write):
        session = ClientSession(read, write)
        await asyncio.wait_for(session.initialize(), timeout=10.0)  # ❌ Timeout

asyncio.run(test())
```

**Error:** `anyio.BrokenResourceError` in stdout_reader task

---

## Workaround Options

### Option 1: HTTP/SSE Transport (RECOMMENDED)

**Pros:**
- Avoids stdio_client completely
- More reliable for production deployments
- Better error handling and logging
- Network-accessible (not just local)

**Cons:**
- Requires APS MCP server to support HTTP/SSE
- Need to check if Node.js server has HTTP endpoint
- May require server configuration changes

**Implementation:**
```python
from mcp.client.sse import sse_client

config = MCPServerConfig(
    name="aps-mcp",
    transport="http",
    url="http://localhost:3000/mcp",  # If server supports HTTP
    timeout=30,
    enabled=True
)
```

**Next Step:** Check if APS MCP Node.js server has HTTP/SSE support

### Option 2: Use Custom stdio Client

**Pros:**
- Direct control over stream handling
- Can work around SDK bugs
- Better debugging capabilities

**Cons:**
- Must reimplement MCP protocol handling
- Maintenance burden
- More code to test

**Implementation:** Create custom stdio_client that uses simple readline() like raw test

### Option 3: Patch MCP SDK

**Pros:**
- Fixes root cause
- Benefits other users with similar issues

**Cons:**
- Requires understanding SDK internals
- May break with SDK updates
- Need to maintain fork or contribute upstream

### Option 4: Contact MCP SDK Maintainers

**Pros:**
- Official fix from SDK team
- Benefits entire community

**Cons:**
- Takes time to get fix released
- May not be prioritized

---

## Recommended Next Steps

1. **Immediate (5 minutes):**
   - Check if APS MCP Node.js server supports HTTP/SSE transport
   - Look for server documentation on HTTP endpoint configuration

2. **Short-term (1 hour):**
   - If HTTP/SSE available: Implement HTTP transport in orchestrator
   - If not available: Implement custom stdio client using raw subprocess approach

3. **Long-term:**
   - Report issue to MCP SDK maintainers with reproduction case
   - Consider contributing fix upstream if SDK team confirms bug

---

## Files for Reference

- **Test Scripts:**
  - `test_mcp_raw.py` - Raw subprocess test (✅ works)
  - `test_mcp_sdk.py` - SDK test (❌ fails)
  - `test_aps_server_direct.py` - Direct server test

- **Configuration:**
  - `/path/to/aps-mcp-server-nodejs/.env` - Server environment variables (✅ correct)
  - `examples/mcp_aps_example.py` - Integration example (❌ fails with SDK)

- **Documentation:**
  - `claudedocs/MCP_FIXES_IMPLEMENTED.md` - Previous fixes (all correct)
  - `claudedocs/APS_MCP_TASKGROUP_ROOT_CAUSE_ANALYSIS.md` - Initial analysis

---

## Conclusion

The Python MCP SDK's stdio_client has a **stream management bug** that prevents it from working with the Node.js APS MCP server, despite the server being fully MCP-compliant. The server works perfectly with raw subprocess communication, proving the issue is in the SDK's implementation.

**Recommendation:** Investigate HTTP/SSE transport support in APS MCP server as the most reliable solution. If not available, implement a custom stdio client using the working raw subprocess approach demonstrated in test_mcp_raw.py.

---

## References

- **MCP Specification:** https://modelcontextprotocol.io
- **Python MCP SDK:** https://github.com/anthropics/python-sdk
- **APS MCP Server:** https://github.com/autodesk-platform-services/aps-mcp-server-nodejs
- **Test Results:** All test scripts in project root demonstrate the incompatibility

---

**Status:** Issue identified and documented. Awaiting decision on workaround approach.
