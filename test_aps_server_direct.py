"""
Direct test of APS MCP server to diagnose connection issues.
"""

import asyncio
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator.mcp import MCPServerConfig, MCPClientManager


async def test_aps_server():
    """Test APS MCP server connection with detailed logging."""

    config = MCPServerConfig(
        name="aps-mcp",
        transport="stdio",
        command="node",
        args=["/Users/ndelvalalvarez/Downloads/PROYECTOS/aps-mcp-server-nodejs/server.js"],
        env={},  # Inherits from parent
        protocol_version="2024-11-05",
        timeout=60,
        enabled=True
    )

    client_manager = MCPClientManager()

    try:
        print("=" * 60)
        print("Testing APS MCP Server Connection")
        print("=" * 60)
        print()

        print("1️⃣ Configuration:")
        print(f"   Command: {config.command}")
        print(f"   Args: {config.args}")
        print(f"   Protocol: {config.protocol_version}")
        print(f"   Timeout: {config.timeout}s")
        print()

        print("2️⃣ Connecting to server...")
        session = await client_manager.get_session(config)
        print("   ✅ Connection successful!")
        print()

        print("3️⃣ Listing tools...")
        tools = await client_manager.list_tools(config)
        print(f"   ✅ Found {len(tools)} tools:")
        for tool in tools:
            print(f"      - {tool['name']}")
        print()

        print("4️⃣ Calling getProjectsTool...")
        result = await client_manager.call_tool(
            config=config,
            tool_name="getProjectsTool",
            arguments={}
        )
        print(f"   ✅ Result type: {type(result)}")
        if hasattr(result, 'content'):
            print(f"   ✅ Content items: {len(result.content)}")
            for item in result.content:
                if hasattr(item, 'text'):
                    text = item.text
                    print(f"   ✅ Text preview ({len(text)} chars): {text[:100]}...")
        print()

        print("=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print()
        print("=" * 60)
        print("❌ Test failed!")
        print("=" * 60)
        print(f"Error: {e}")
        print()

        import traceback
        print("Traceback:")
        traceback.print_exc()

    finally:
        await client_manager.cleanup()
        print("\n🧹 Cleanup complete")


if __name__ == "__main__":
    asyncio.run(test_aps_server())
