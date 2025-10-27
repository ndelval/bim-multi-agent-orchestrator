# MCP .NET Server Connection Issue - Root Cause Analysis

## Problem Statement

The MCP Python client fails to connect to the Autodesk AEC Data Model MCP server (.NET implementation) with either timeout or protocol negotiation errors, despite the server responding correctly to manual JSON-RPC requests.

## Evidence Gathered

### 1. Direct Subprocess Test (SUCCESSFUL)
```python
# Manual JSON-RPC request sent directly to dotnet process
# Protocol: "2024-11-05"
# Result: ✅ SUCCESS - Server responded correctly
```

**Server Response**:
```json
{
  "jsonrpc":"2.0",
  "id":1,
  "result":{
    "protocolVersion":"2024-11-05",
    "capabilities":{"tools":{"listChanged":true}},
    "serverInfo":{"name":"mcp-server-aecdm","version":"1.0.0.0"}
  }
}
```

### 2. MCP SDK Client Test (FAILED)
```
Error: SessionMessage(...) with protocolVersion: '2025-06-18'
```

**MCP SDK Version**: 1.12.4
**Protocol Version Sent**: 2025-06-18
**Server Expects**: 2024-11-05

## Root Cause

**PRIMARY**: Protocol version mismatch between MCP Python SDK (1.12.4) and .NET MCP server implementation.

- **MCP Python SDK 1.12.4** uses protocol version `2025-06-18` (from `mcp/types.py:26`)
- **.NET MCP Server** implements protocol version `2024-11-05`
- The MCP protocol negotiation fails because the versions are incompatible

**SECONDARY**: Context manager lifecycle issues in original implementation (now fixed).

- Original code called `__aenter__()` without proper `__aexit__()` handling
- This caused subprocess cleanup issues
- **Status**: ✅ FIXED in current implementation

## Why Manual Test Works But SDK Fails

1. **Manual Test**: We explicitly sent `"protocolVersion": "2024-11-05"` matching what the server expects
2. **SDK Test**: The MCP SDK hard-codes `LATEST_PROTOCOL_VERSION = "2025-06-18"` and sends it automatically

## Solutions

### Option 1: Downgrade MCP SDK (RECOMMENDED)
```bash
uv pip install "mcp>=1.0.0,<1.12.0"
```
Find a version that uses protocol version `2024-11-05`.

### Option 2: Upgrade .NET Server
Clone and use the latest version from:
```
https://github.com/joaomartins-callmejohn/aps-aecdm-mcp-dotnet
```
Check if a newer version supports protocol `2025-06-18`.

### Option 3: Custom Protocol Negotiation (COMPLEX)
Modify `ClientSession.initialize()` to negotiate protocol version dynamically.

## Implementation Status

### ✅ Fixed Issues
1. Context manager lifecycle handling
2. Proper subprocess cleanup
3. Timeout handling with configurable duration
4. Better error messages with protocol version information

### ⏳ Remaining Issue
- Protocol version mismatch requires SDK version adjustment or server upgrade

## Testing Strategy

### Test 1: Verify Protocol Version Compatibility
```python
# Check what protocol version the SDK will use
from mcp.types import LATEST_PROTOCOL_VERSION
print(f"SDK Protocol: {LATEST_PROTOCOL_VERSION}")
```

### Test 2: Test with Older MCP SDK Version
```bash
uv pip install "mcp==1.0.0"  # or another version
python examples/test_autodesk_aec_mcp.py
```

### Test 3: Check .NET Server Version
```bash
cd mcp-servers/aps-aecdm-mcp-dotnet
git log --oneline -5  # Check recent updates
git pull  # Get latest version
```

## Recommended Next Steps

1. **Immediate**: Check MCP SDK changelog for protocol version changes
2. **Short-term**: Test with MCP SDK versions that use protocol `2024-11-05`
3. **Long-term**: Update constraint in `pyproject.toml` to compatible version range

## Files Modified

- `orchestrator/mcp/client_manager.py`: Context manager lifecycle fixes, timeout handling
- `orchestrator/mcp/stdio_custom.py`: Custom stdio implementation (experimental)
- `examples/debug_dotnet_mcp.py`: Diagnostic script proving server works

## References

- MCP Protocol Versions: https://modelcontextprotocol.io/specification
- .NET Server Repo: https://github.com/joaomartins-callmejohn/aps-aecdm-mcp-dotnet
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
