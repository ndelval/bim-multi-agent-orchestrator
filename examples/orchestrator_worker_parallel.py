"""
Orchestrator–Worker example with Autonomous Workflow and Parallelization (PraisonAI Agents)

What this shows:
- Orchestrator agent that delegates and coordinates specialized workers
- Autonomous workflow routing with conditional decisions
- Parallel execution of independent tasks (async)

Requirements:
- Python 3.10+
- pip install: praisonaiagents duckduckgo-search
- An LLM provider configured via env (e.g., OPENAI_API_KEY) or your preferred LiteLLM-compatible provider

Run:
  python examples/orchestrator_worker_parallel.py

Notes:
- If you want true parallel execution, we call `astart()` with tasks that have `async_execution=True` and no dependencies.
- The engine will schedule independent tasks concurrently and then join for dependent ones.
"""

from typing import List

from praisonaiagents import Agent, Task, PraisonAIAgents
from praisonaiagents.tools import duckduckgo


def build_orchestrator_system() -> PraisonAIAgents:
    # Orchestrator / Manager
    orchestrator = Agent(
        name="Orchestrator",
        role="AI Orchestrator",
        goal=(
            "Plan, route, and coordinate work among specialized agents "
            "to produce a high‑quality final deliverable."
        ),
        backstory=(
            "You are a seasoned project orchestrator that breaks down goals into sub‑tasks, "
            "assigns them to the right specialists, monitors progress, adapts to feedback, "
            "and ensures timely, high‑quality outcomes."
        ),
        instructions=(
            "Analyze objectives, propose a minimal viable plan, assign parallelizable tasks, "
            "monitor results, and request revisions when quality risks are detected."
        ),
    )

    # Workers
    researcher = Agent(
        name="Researcher",
        role="Web Research Specialist",
        goal="Gather up‑to‑date, sourced information",
        backstory="Expert in web research and summarization.",
        tools=[duckduckgo],
        instructions="Use web search to collect reliable, relevant information with sources.",
    )

    planner = Agent(
        name="Planner",
        role="Solution Planner",
        goal="Transform goals and research into an actionable plan",
        backstory="You create pragmatic plans that balance speed and quality.",
        instructions=(
            "Propose a concise plan with steps, owners, and acceptance criteria. "
            "Prefer parallelizable steps where safe."
        ),
    )

    implementer = Agent(
        name="Implementer",
        role="Prototype Builder",
        goal="Create a simple proof‑of‑concept based on the plan",
        backstory="You build minimal prototypes quickly and document trade‑offs.",
        instructions=(
            "Implement the simplest viable approach that satisfies the plan’s acceptance criteria."
        ),
    )

    tester = Agent(
        name="Tester",
        role="QA Specialist",
        goal="Validate functionality and quality",
        backstory="You design lean checks to validate core functionality.",
        instructions="Test critical paths; report defects with clear reproduction steps.",
    )

    writer = Agent(
        name="Writer",
        role="Technical Writer",
        goal="Produce a crisp executive summary",
        backstory="You synthesize complex outputs into clear narratives.",
        instructions=(
            "Create a concise report: objective, approach, key findings, limitations, next steps."
        ),
    )

    # Tasks
    # Fan‑out: research and planning can run in parallel (async_execution=True, no dependencies)
    research_task = Task(
        name="research_task",
        description=(
            "Research current best practices for building agentic orchestrator‑worker systems with "
            "autonomous workflows and parallel task execution. Cite sources."
        ),
        expected_output="Bullet list of best practices with 3–5 citations",
        agent=researcher,
        async_execution=True,
        is_start=True,
    )

    plan_task = Task(
        name="plan_task",
        description=(
            "Based on the goal, propose a minimal plan to implement an orchestrator with parallel workers. "
            "Include steps, owners, and acceptance criteria."
        ),
        expected_output="Plan with steps, ownership, criteria, and risks",
        agent=planner,
        async_execution=True,
        is_start=True,
    )

    # Join: implementation depends on both research and plan
    implement_task = Task(
        name="implement_task",
        description=(
            "Implement a minimal prototype design (in prose) following the plan and informed by research. "
            "Describe modules, responsibilities, and interfaces."
        ),
        expected_output="Prototype design notes covering modules and interfaces",
        agent=implementer,
        context=[research_task, plan_task],
        next_tasks=["test_task"],
    )

    test_task = Task(
        name="test_task",
        description=(
            "Propose a lean test strategy to validate the prototype design. List test cases and expected outcomes."
        ),
        expected_output="Test plan with critical cases and pass criteria",
        agent=tester,
        context=[implement_task],
        next_tasks=["writeup_task"],
    )

    writeup_task = Task(
        name="writeup_task",
        description=(
            "Draft an executive summary combining the plan, design, and test strategy."
        ),
        expected_output="Executive summary with objective, approach, findings, next steps",
        agent=writer,
        context=[research_task, plan_task, implement_task, test_task],
        next_tasks=["review_task"],
    )

    # Autonomous decision: orchestrator reviews and routes
    review_task = Task(
        name="review_task",
        description=(
            "Review all outputs for quality and coherence. If gaps exist, decide whether to request "
            "revision of implementation or tests, else approve."
        ),
        expected_output=(
            "Decision: approved | needs_revision_implementation | needs_revision_tests with actionable feedback"
        ),
        agent=orchestrator,
        context=[research_task, plan_task, implement_task, test_task, writeup_task],
        task_type="decision",
        condition={
            # Map decision labels to next tasks for autonomous routing
            "approved": ["finalize_task"],
            "needs_revision_implementation": ["implement_task"],
            "needs_revision_tests": ["test_task"],
        },
    )

    finalize_task = Task(
        name="finalize_task",
        description=(
            "Produce final handoff notes: what was done, quality level, and recommended follow‑ups."
        ),
        expected_output="Final handoff summary with follow‑ups",
        agent=orchestrator,
        context=[review_task, writeup_task],
    )

    # Recommended memory configuration (Local RAG + embeddings)
    # Alternatives:
    # - MongoDB (persistent, scalable): set provider="mongodb" and connection string
    # - Mem0 Graph (relationships + vector): provider="mem0" with graph_store
    memory_config = {
        "provider": "rag",  # "rag" | "mongodb" | "mem0"
        "use_embedding": True,  # Enable vector retrieval
        "short_db": ".praison/memory/short.db",
        "long_db": ".praison/memory/long.db",
        "rag_db_path": ".praison/memory/chroma_db",
        "embedder": {
            "provider": "openai",
            "config": {"model": "text-embedding-3-large"},
        },
    }

    # Example: MongoDB provider (uncomment and set credentials)
    # memory_config = {
    #     "provider": "mongodb",
    #     "use_embedding": True,
    #     "config": {
    #         "connection_string": "mongodb+srv://<user>:<pass>@<cluster>/?retryWrites=true&w=majority",
    #         "database": "praisonai",
    #         "use_vector_search": True,
    #         "max_pool_size": 100
    #     },
    #     "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-large"}}
    # }

    # Example: Mem0 with Graph store (requires `pip install mem0ai[graph]`)
    # memory_config = {
    #     "provider": "mem0",
    #     "use_embedding": True,
    #     "config": {
    #         "graph_store": {
    #             "provider": "neo4j",
    #             "config": {
    #                 "url": "neo4j+s://<host>",
    #                 "username": "neo4j",
    #                 "password": "<password>"
    #             }
    #         },
    #         # Optional vector store (e.g., Qdrant)
    #         # "vector_store": {"provider": "qdrant", "config": {"host": "localhost", "port": 6333}},
    #         "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-large"}},
    #         "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini"}}
    #     }
    # }

    system = PraisonAIAgents(
        agents=[orchestrator, researcher, planner, implementer, tester, writer],
        tasks=[
            research_task,
            plan_task,
            implement_task,
            test_task,
            writeup_task,
            review_task,
            finalize_task,
        ],
        # Use workflow to enable decision routing; async tasks will run in parallel when independent
        process="workflow",
        verbose=1,
        max_iter=8,
        name="OrchestratorWorkerParallel",
        memory=True,
        memory_config=memory_config,
        # embedder can also be passed at the top level to override the memory config model
        embedder={"provider": "openai", "config": {"model": "text-embedding-3-large"}},
        user_id="demo-user",
    )

    return system


def main():
    print("hola")
    system = build_orchestrator_system()

    # Prefer async to enable parallel execution of independent tasks
    try:
        import asyncio

        result = asyncio.run(system.astart())  # runs parallelizable tasks concurrently
    except AttributeError:
        # Fallback if astart is unavailable
        result = system.start()

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
