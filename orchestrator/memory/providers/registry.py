"""Simple registry mapping provider names to memory provider classes."""

from __future__ import annotations

from typing import Dict, Optional, Type

from .base import IMemoryProvider
from .hybrid_provider import HybridRAGMemoryProvider
from .mem0_provider import Mem0MemoryProvider
from .rag_provider import SimpleMemoryProvider


class MemoryProviderRegistry:
    """Compatible registry interface for legacy CLI imports."""

    _PROVIDERS: Dict[str, Type[IMemoryProvider]] = {
        "rag": SimpleMemoryProvider,
        "mem0": Mem0MemoryProvider,
        "hybrid": HybridRAGMemoryProvider,
    }

    @classmethod
    def get_provider(cls, name: Optional[str]) -> Optional[Type[IMemoryProvider]]:
        if not name:
            return None
        return cls._PROVIDERS.get(name.lower())

    @classmethod
    def register_provider(cls, name: str, provider_cls: Type[IMemoryProvider]) -> None:
        cls._PROVIDERS[name.lower()] = provider_cls
