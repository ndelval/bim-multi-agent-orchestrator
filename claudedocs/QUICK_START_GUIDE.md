# Quick Start Guide - AI Agents Unified Framework

## 🎯 Get Started in 5 Minutes

This guide will get you up and running with the AI Agents Unified Framework in the fastest way possible.

---

## Step 1: Installation (2 minutes)

### Basic Setup

```bash
# Clone repository
git clone <repository-url>
cd PruebasMultiAgent

# Install with uv (recommended - faster)
pip install uv
uv pip install -e .[all-basic]

# Create .env file
cat > .env << 'EOF'
OPENAI_API_KEY=your_key_here
ORCH_MEMORY_PROVIDER=rag
EOF
```

**Note**: Using `rag` provider requires no additional dependencies (Neo4j, Qdrant). Perfect for quick starts!

---

## Step 2: First Chat (1 minute)

```bash
# Start interactive chat
python -m orchestrator.cli chat --memory-provider rag

# Example conversation:
# You: What are the best practices for API security?
# System: [Analyzes query with multiple agents]
# System: [Returns comprehensive answer with citations]
```

---

## Step 3: Programmatic Usage (2 minutes)

Create `my_first_workflow.py`:

```python
from orchestrator import Orchestrator, OrchestratorConfig
from orchestrator.core.config import AgentConfig, TaskConfig, MemoryConfig

# 1. Create configuration
config = OrchestratorConfig(
    name="MyFirstWorkflow",
    process="sequential",
    memory=MemoryConfig(provider="rag")
)

# 2. Add a simple agent
config.agents.append(AgentConfig(
    name="Researcher",
    role="Research Specialist",
    goal="Gather information on any topic",
    backstory="Expert researcher with web search skills",
    instructions="Search web and provide comprehensive answers",
    tools=[],  # No external tools needed for basic usage
    llm="gpt-4o-mini"
))

# 3. Add a task
config.tasks.append(TaskConfig(
    name="research_task",
    description="Research Python best practices for 2024",
    expected_output="Comprehensive research findings",
    agent="Researcher"
))

# 4. Execute
orchestrator = Orchestrator(config)
result = orchestrator.run_sync()

print(result)
```

Run it:

```bash
python my_first_workflow.py
```

---

## What's Next?

### Add Memory (Optional but Recommended)

Enable hybrid memory for better retrieval:

```bash
# Install Neo4j with Docker
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest

# Update .env
ORCH_MEMORY_PROVIDER=hybrid
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Now use hybrid provider
python -m orchestrator.cli chat --memory-provider hybrid
```

### Add Web Search Tools

```python
from praisonaiagents.tools.duckduckgo import duckduckgo_tool
from praisonaiagents.tools.wikipedia import wikipedia_tool

config.agents[0].tools = [duckduckgo_tool, wikipedia_tool]
```

### Add More Agents

```python
config.agents.extend([
    AgentConfig(
        name="Analyst",
        role="Data Analyst",
        goal="Analyze research findings",
        backstory="Expert in data analysis",
        instructions="Analyze and summarize findings",
        tools=[],
        llm="gpt-4o-mini"
    ),
    AgentConfig(
        name="Writer",
        role="Technical Writer",
        goal="Create clear documentation",
        backstory="Professional technical writer",
        instructions="Write clear, concise summaries",
        tools=[],
        llm="gpt-4o-mini"
    )
])

# Add tasks with dependencies
config.tasks.extend([
    TaskConfig(
        name="analysis_task",
        description="Analyze research findings",
        expected_output="Analysis report",
        agent="Analyst",
        context=["research_task"]  # Depends on research
    ),
    TaskConfig(
        name="writing_task",
        description="Write final report",
        expected_output="Professional report",
        agent="Writer",
        context=["research_task", "analysis_task"]  # Depends on both
    )
])
```

### Enable Parallel Execution

```python
config.process = "workflow"  # Use DAG-based workflow

# Mark tasks as parallelizable
config.tasks[0].async_execution = True
config.tasks[0].is_start = True

# Add another parallel start task
config.tasks.append(TaskConfig(
    name="alternative_research",
    description="Research alternative approaches",
    expected_output="Alternative findings",
    agent="Researcher",
    is_start=True,  # Will run in parallel with research_task
    async_execution=True
))
```

---

## Common Patterns

### Pattern 1: Research → Analysis → Report

```python
agents = [
    ("Researcher", "Research Specialist", "Gather information"),
    ("Analyst", "Data Analyst", "Analyze findings"),
    ("Writer", "Technical Writer", "Create reports")
]

tasks = [
    ("research", "Researcher", "Research topic X", None),
    ("analysis", "Analyst", "Analyze research", ["research"]),
    ("report", "Writer", "Write report", ["research", "analysis"])
]
```

### Pattern 2: Parallel Research → Synthesis

```python
# Two researchers work in parallel
tasks = [
    ("web_research", "WebResearcher", "Search web", None, True, True),
    ("academic_research", "AcademicResearcher", "Search papers", None, True, True),
    ("synthesis", "Synthesizer", "Combine findings", ["web_research", "academic_research"])
]
```

### Pattern 3: Conditional Routing

```python
TaskConfig(
    name="risk_assessment",
    agent="SecurityAnalyst",
    task_type="decision",
    condition={
        "high_risk": ["emergency_response"],
        "medium_risk": ["standard_review"],
        "low_risk": ["documentation_only"]
    }
)
```

---

## Troubleshooting

### "No module named 'orchestrator'"

```bash
# Make sure you're in the project directory
cd PruebasMultiAgent

# Reinstall
pip install -e .
```

### "Memory provider not initialized"

```bash
# Use simple RAG provider
python -m orchestrator.cli chat --memory-provider rag

# Or check Neo4j is running
docker ps | grep neo4j
```

### "OPENAI_API_KEY not found"

```bash
# Check .env file exists
cat .env

# Or set directly
export OPENAI_API_KEY=your_key_here
```

---

## Next Steps

1. **Read Full Documentation**
   - [Main README](../README.md) - Complete project overview
   - [Orchestrator Architecture](ORCHESTRATOR_ARCHITECTURE.md) - Deep dive

2. **Try Examples**
   ```bash
   # PraisonAI examples
   python PraisonAI/src/praisonai-agents/test_multi_agents.py

   # Orchestrator examples
   python orchestrator/factories/tests/test_agent_factory.py
   ```

3. **Explore Advanced Features**
   - Tree-of-Thought planning
   - GraphRAG memory retrieval
   - LangGraph integration
   - Custom agent templates

4. **Join Community**
   - Create issues for bugs
   - Use Discussions for questions
   - Contribute improvements

---

## Quick Reference Card

### Environment Variables

```bash
OPENAI_API_KEY=sk-...           # Required
ORCH_MEMORY_PROVIDER=rag        # rag|hybrid|mem0
NEO4J_URL=bolt://localhost:7687 # For hybrid/mem0
NEO4J_USER=neo4j                # For hybrid/mem0
NEO4J_PASSWORD=password         # For hybrid/mem0
```

### CLI Commands

```bash
# Chat
python -m orchestrator.cli chat --memory-provider rag

# Memory info
python -m orchestrator.cli memory-info --memory-provider hybrid

# With verbose logging
python -m orchestrator.cli chat --verbose
```

### Python API

```python
# Import
from orchestrator import Orchestrator, OrchestratorConfig
from orchestrator.core.config import AgentConfig, TaskConfig

# Create
config = OrchestratorConfig(name="MyWorkflow")
config.agents.append(AgentConfig(...))
config.tasks.append(TaskConfig(...))

# Execute
orchestrator = Orchestrator(config)
result = orchestrator.run_sync()
```

---

**You're now ready to build production AI agent systems!** 🚀

For more examples and advanced patterns, see the [full documentation](../README.md).
