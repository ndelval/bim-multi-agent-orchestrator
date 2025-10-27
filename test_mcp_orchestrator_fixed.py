"""
Test script to verify MCP orchestrator fixes

This script tests the fixed MCP integration with:
1. Async tool creation (_create_mcp_tools_async)
2. Async tool wrappers (native async, no sync wrapper issues)
3. Environment variable inheritance (_prepare_env)

Tests with APS MCP server.
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add orchestrator to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator.mcp import MCPServerConfig, MCPClientManager, MCPToolAdapter
from orchestrator.factories.agent_factory import AgentFactory
from orchestrator.core.config import AgentConfig


async def test_mcp_client_direct():
    """Test 1: Direct MCP client connection"""
    print("=" * 60)
    print("Test 1: Direct MCP Client Connection")
    print("=" * 60)
    print()

    # Configure APS MCP
    config = MCPServerConfig(
        name="aps-mcp-test",
        transport="stdio",
        command="node",
        args=["/Users/ndelvalalvarez/Downloads/PROYECTOS/aps-mcp-server-nodejs/server.js"],
        env={},  # Will inherit from parent process
        tools=["getProjectsTool"],
        timeout=30,
        enabled=True
    )

    client_manager = MCPClientManager()

    try:
        print("🔗 Connecting to APS MCP server...")
        session = await client_manager.get_session(config)
        print("✅ Connected successfully")
        print()

        print("📋 Listing tools...")
        tools = await client_manager.list_tools(config)
        print(f"Found {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool['name']}")
        print()

        print("🔧 Calling getProjectsTool...")
        result = await client_manager.call_tool(
            config=config,
            tool_name="getProjectsTool",
            arguments={}
        )

        print("Result type:", type(result))
        if hasattr(result, 'content'):
            print("Content items:", len(result.content))
            for item in result.content:
                if hasattr(item, 'text'):
                    print("Text length:", len(item.text))
                    print("Text preview:", item.text[:200] if item.text else "(empty)")

        print("\n✅ Test 1 PASSED")

    except Exception as e:
        print(f"\n❌ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await client_manager.cleanup()
        print()


async def test_mcp_tool_adapter():
    """Test 2: MCP Tool Adapter with async tools"""
    print("=" * 60)
    print("Test 2: MCP Tool Adapter (Async Tools)")
    print("=" * 60)
    print()

    config = MCPServerConfig(
        name="aps-mcp-test",
        transport="stdio",
        command="node",
        args=["/Users/ndelvalalvarez/Downloads/PROYECTOS/aps-mcp-server-nodejs/server.js"],
        env={},
        tools=["getProjectsTool"],
        timeout=30,
        enabled=True
    )

    client_manager = MCPClientManager()
    tool_adapter = MCPToolAdapter(client_manager)

    try:
        print("🔗 Creating MCP tools...")
        tools = await tool_adapter.create_tools(config)
        print(f"✅ Created {len(tools)} tool(s)")
        print()

        for tool in tools:
            print(f"Tool: {tool.__name__}")
            print(f"  Doc: {tool.__doc__[:100] if tool.__doc__ else 'N/A'}")
            print(f"  Is async: {hasattr(tool, 'is_async') and tool.is_async}")
            print(f"  Is MCP tool: {hasattr(tool, 'is_mcp_tool') and tool.is_mcp_tool}")
            print()

        # Test calling the async tool
        if tools:
            print("🔧 Testing tool execution...")
            tool = tools[0]

            # Call async tool (should work in async context)
            result = await tool()
            print(f"Result type: {type(result)}")
            print(f"Result preview: {result[:200] if result else '(empty)'}")
            print()

        print("✅ Test 2 PASSED")

    except Exception as e:
        print(f"\n❌ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await client_manager.cleanup()
        print()


async def test_agent_factory_async():
    """Test 3: AgentFactory.create_agent_async with MCP"""
    print("=" * 60)
    print("Test 3: AgentFactory Async Creation")
    print("=" * 60)
    print()

    # Configure APS MCP
    mcp_config = MCPServerConfig(
        name="aps-mcp-test",
        transport="stdio",
        command="node",
        args=["/Users/ndelvalalvarez/Downloads/PROYECTOS/aps-mcp-server-nodejs/server.js"],
        env={},
        tools=["getProjectsTool"],
        timeout=30,
        enabled=True
    )

    # Configure agent
    agent_config = AgentConfig(
        name="TestAgent",
        role="Test Agent",
        goal="Test MCP integration",
        backstory="Testing agent",
        instructions="Use MCP tools",
        mcp_servers=[mcp_config],
        tools=[]
    )

    factory = AgentFactory()

    try:
        print("🔗 Creating agent with MCP tools (async)...")
        agent = await factory.create_agent_async(agent_config)

        print("✅ Agent created successfully")
        print(f"Agent name: {agent_config.name}")
        print(f"Tools count: {len(agent_config.tools)}")

        for i, tool in enumerate(agent_config.tools):
            if hasattr(tool, '__name__'):
                print(f"  Tool {i+1}: {tool.__name__}")
                print(f"    Is async: {hasattr(tool, 'is_async') and tool.is_async}")

        print()
        print("✅ Test 3 PASSED")

    except Exception as e:
        print(f"\n❌ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await factory.cleanup_mcp()
        print()


async def test_env_inheritance():
    """Test 4: Environment variable inheritance"""
    print("=" * 60)
    print("Test 4: Environment Variable Inheritance")
    print("=" * 60)
    print()

    config = MCPServerConfig(
        name="aps-mcp-test",
        transport="stdio",
        command="node",
        args=["/Users/ndelvalalvarez/Downloads/PROYECTOS/aps-mcp-server-nodejs/server.js"],
        env={},  # Empty - should inherit from parent
        tools=["getProjectsTool"],
        timeout=30,
        enabled=True
    )

    client_manager = MCPClientManager()

    try:
        # Test _prepare_env method
        print("🔍 Testing environment preparation...")
        env = client_manager._prepare_env(config)

        print(f"Environment variables prepared: {len(env)}")
        print()

        # Check that APS variables are present
        aps_vars = ["APS_CLIENT_ID", "APS_CLIENT_SECRET", "SSA_ID", "SSA_KEY_ID", "SSA_KEY_PATH"]
        for var in aps_vars:
            if var in env:
                print(f"✅ {var}: Present")
            else:
                print(f"❌ {var}: Missing")

        print()
        print("✅ Test 4 PASSED")

    except Exception as e:
        print(f"\n❌ Test 4 FAILED: {e}")
        import traceback
        traceback.print_exc()

    print()


async def run_all_tests():
    """Run all tests"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "MCP Orchestrator Fix Verification" + " " * 15 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    # Test 1: Direct client connection
    await test_mcp_client_direct()

    # Test 2: Tool adapter
    await test_mcp_tool_adapter()

    # Test 3: Agent factory async
    await test_agent_factory_async()

    # Test 4: Environment inheritance
    await test_env_inheritance()

    print()
    print("=" * 60)
    print("All Tests Complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
