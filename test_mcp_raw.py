"""
Raw test of MCP protocol communication to understand initialization failure.
"""

import asyncio
import json
import sys


async def test_raw_mcp():
    """Test raw MCP protocol communication."""

    print("=" * 60)
    print("Testing Raw MCP Protocol")
    print("=" * 60)
    print()

    # Start Node.js server
    print("1️⃣ Starting Node.js server subprocess...")
    process = await asyncio.create_subprocess_exec(
        "node",
        "/Users/ndelvalalvarez/Downloads/PROYECTOS/aps-mcp-server-nodejs/server.js",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    print(f"   ✅ Process started (PID: {process.pid})")
    print()

    # Send initialize request
    print("2️⃣ Sending initialize request...")
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0"
            }
        }
    }

    request_str = json.dumps(init_request) + "\n"
    print(f"   Request: {request_str[:100]}...")

    try:
        process.stdin.write(request_str.encode())
        await process.stdin.drain()
        print("   ✅ Request sent")
        print()

        # Read response with timeout
        print("3️⃣ Waiting for response (10s timeout)...")
        try:
            response_bytes = await asyncio.wait_for(
                process.stdout.readline(),
                timeout=10.0
            )
            response = response_bytes.decode()
            print(f"   ✅ Response received ({len(response)} bytes)")
            print(f"   Response: {response[:200]}...")
            print()

            # Parse JSON
            response_data = json.loads(response)
            print(f"   ✅ Protocol version: {response_data['result']['protocolVersion']}")
            print(f"   ✅ Server: {response_data['result']['serverInfo']['name']}")
            print()

            print("=" * 60)
            print("✅ Raw MCP protocol test passed!")
            print("=" * 60)

        except asyncio.TimeoutError:
            print("   ❌ Timeout waiting for response")

            # Check stderr
            print("\n4️⃣ Checking stderr...")
            stderr_data = await process.stderr.read()
            if stderr_data:
                print(f"   Stderr: {stderr_data.decode()}")
            else:
                print("   No stderr output")

    except Exception as e:
        print(f"   ❌ Error: {e}")

    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        process.kill()
        await process.wait()
        print("   ✅ Process terminated")


if __name__ == "__main__":
    asyncio.run(test_raw_mcp())
