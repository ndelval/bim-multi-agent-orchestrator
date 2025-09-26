"""Utility to resynchronize graph nodes/relations from hybrid storage."""

from __future__ import annotations

import argparse

from orchestrator.cli.main import resolve_memory_config
from orchestrator.memory.memory_manager import MemoryManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resync Neo4j graph with hybrid memory metadata")
    parser.add_argument(
        "--memory-provider",
        choices=["mem0", "hybrid"],
        default="hybrid",
        help="Memory provider to use (default hybrid)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    memory_config = resolve_memory_config(args.memory_provider)
    manager = MemoryManager(memory_config)

    count = manager.sync_graph()
    print(f"✅ Resynchronized {count} chunks into the graph")


if __name__ == "__main__":
    main()

