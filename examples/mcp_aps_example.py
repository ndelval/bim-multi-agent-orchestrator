"""
Example: Using Autodesk Platform Services (APS) MCP Server with Orchestrator

This example demonstrates how to configure and use the APS MCP server
to access Autodesk Construction Cloud (ACC) projects and data.

Prerequisites:
- APS MCP server installed: git clone https://github.com/autodesk-platform-services/aps-mcp-server-nodejs.git
- Node.js installed
- APS credentials configured in environment

Environment Variables Required:
- APS_CLIENT_ID: Your APS application client ID
- APS_CLIENT_SECRET: Your APS application client secret
- SSA_ID: Your service account ID
- SSA_KEY_ID: Your private key ID
- SSA_KEY_PATH: Path to your .pem private key file
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import Orchestrator, OrchestratorConfig
from orchestrator.core.config import AgentConfig, TaskConfig
from orchestrator.mcp import MCPServerConfig
from dotenv import load_dotenv

load_dotenv()


async def main():
    """Main example function."""

    print("=" * 60)
    print("APS MCP Integration Example")
    print("=" * 60)
    print()

    # 1. Verify environment variables
    required_vars = [
        "APS_CLIENT_ID",
        "APS_CLIENT_SECRET",
        "SSA_ID",
        "SSA_KEY_ID",
        "SSA_KEY_PATH",
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print()
        print("Please set them in your .env file or environment:")
        for var in missing_vars:
            print(f"  export {var}=your_value_here")
        return

    print("✅ All required environment variables are set")
    print()

    # 2. Configure APS MCP Server
    # Note: env vars are inherited from parent process and can be overridden
    # Important: Specify protocol_version to match the server's version
    aps_mcp_config = MCPServerConfig(
        name="aps-mcp",
        transport="stdio",
        command="node",
        args=[
            "/Users/ndelvalalvarez/Downloads/PROYECTOS/aps-mcp-server-nodejs/server.js"
        ],
        env={
            # These override parent environment if needed
            # Leave empty to inherit all env vars from parent process
        },
        tools=[
            "getProjectsTool",
            "getFolderContentsTool",
            "getIssueTypesTool",
            "getIssuesTool",
        ],
        protocol_version="2024-11-05",  # Match Node.js server version
        timeout=30,
        enabled=True,
    )

    print(f"📋 Configured APS MCP Server:")
    print(f"  Name: {aps_mcp_config.name}")
    print(f"  Transport: {aps_mcp_config.transport}")
    print(f"  Command: {aps_mcp_config.command}")
    print(f"  Tools: {', '.join(aps_mcp_config.tools)}")
    print()

    # 3. Configure Agent with APS MCP Tools
    agent_config = AgentConfig(
        name="APS_Agent",
        role="Autodesk Construction Cloud Specialist",
        goal="Access and analyze ACC project data",
        backstory="Expert in Autodesk Platform Services with deep knowledge of ACC",
        instructions="Use APS MCP tools to retrieve project information",
        mcp_servers=[aps_mcp_config],  # Attach MCP server to agent
        tools=[],  # MCP tools will be added automatically
        llm="gpt-4o-mini",
    )

    # 4. Configure Task
    task_config = TaskConfig(
        name="list_projects",
        description="List all available ACC projects and provide a summary",
        expected_output="Summary of ACC projects with IDs, names, and count",
        agent="APS_Agent",
    )

    # 5. Create Orchestrator
    config = OrchestratorConfig(
        name="APS_MCP_Demo", agents=[agent_config], tasks=[task_config]
    )

    print("🔧 Creating orchestrator...")
    orchestrator = Orchestrator(config)

    # 6. Initialize agent with MCP tools (async)
    print("🔗 Connecting to APS MCP server and initializing tools...")
    try:
        agent = await orchestrator.agent_factory.create_agent_async(agent_config)

        print("✅ APS MCP Integration Ready")
        print(f"\nAgent '{agent_config.name}' has been initialized")
        print(f"MCP tools available: {len(aps_mcp_config.tools)}")
        for tool_name in aps_mcp_config.tools:
            print(f"  ✓ {tool_name}")
        print()

        # 7. Run orchestrator
        print("🚀 Running orchestrator workflow...")
        print("-" * 60)
        result = await orchestrator.run()
        print("-" * 60)

        print("\n📊 Result:")
        print(result)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # 8. Cleanup MCP connections
        print("\n🧹 Cleaning up MCP connections...")
        await orchestrator.agent_factory.cleanup_mcp()
        print("✅ Cleanup complete")


if __name__ == "__main__":
    asyncio.run(main())
