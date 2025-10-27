"""
Test MCP SDK ClientSession to diagnose initialization failure.
"""

import asyncio
import logging
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)


async def test_mcp_sdk():
    """Test MCP SDK ClientSession initialization."""

    print("=" * 60)
    print("Testing MCP SDK ClientSession")
    print("=" * 60)
    print()

    # Configure server
    server_params = StdioServerParameters(
        command="node",
        args=["/Users/ndelvalalvarez/Downloads/PROYECTOS/aps-mcp-server-nodejs/server.js"],
        env=None  # Inherit from parent
    )

    print("1️⃣ Configuration:")
    print(f"   Command: {server_params.command}")
    print(f"   Args: {server_params.args}")
    print()

    print("2️⃣ Creating stdio client...")
    try:
        async with stdio_client(server_params) as (read, write):
            print("   ✅ stdio_client context manager entered")
            print()

            print("3️⃣ Creating ClientSession...")
            session = ClientSession(read, write)
            print("   ✅ ClientSession created")
            print()

            print("4️⃣ Calling session.initialize() with 10s timeout...")
            try:
                result = await asyncio.wait_for(session.initialize(), timeout=10.0)
                print("   ✅ Initialization successful!")
                print(f"   Result: {result}")
                print()

                print("5️⃣ Listing tools...")
                tools = await session.list_tools()
                print(f"   ✅ Found {len(tools.tools)} tools")
                for tool in tools.tools:
                    print(f"      - {tool.name}")
                print()

                print("=" * 60)
                print("✅ MCP SDK test passed!")
                print("=" * 60)

            except asyncio.TimeoutError:
                print("   ❌ Timeout during initialization")
                print()

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_mcp_sdk())
