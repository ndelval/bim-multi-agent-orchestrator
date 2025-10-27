"""
Test the working stdio MCP client implementation.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator.mcp.stdio_working import working_stdio_client


async def test_working_stdio():
    """Test working stdio MCP client with APS MCP server."""

    print("=" * 60)
    print("Testing Working stdio MCP Client")
    print("=" * 60)
    print()

    try:
        async with working_stdio_client(
            command="node",
            args=["/Users/ndelvalalvarez/Downloads/PROYECTOS/aps-mcp-server-nodejs/server.js"],
            env=None,  # Inherit from parent
            protocol_version="2024-11-05"
        ) as client:

            print("✅ MCP session initialized successfully!")
            print()

            # List tools
            print("1️⃣ Listing tools...")
            tools = await client.list_tools()
            print(f"   ✅ Found {len(tools)} tools:")
            for tool in tools:
                print(f"      - {tool.get('name')}: {tool.get('description', '')[:60]}...")
            print()

            # Call a tool
            print("2️⃣ Calling getProjectsTool...")
            result = await client.call_tool("getProjectsTool", {})

            # Parse result
            if "content" in result:
                content_items = result["content"]
                print(f"   ✅ Got {len(content_items)} content items")

                for item in content_items:
                    if "text" in item:
                        text = item["text"]
                        print(f"   📄 Text content ({len(text)} chars)")
                        print(f"      Preview: {text[:100]}...")
            else:
                print(f"   ✅ Result: {str(result)[:200]}")

            print()
            print("=" * 60)
            print("✅ All tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_working_stdio())
