"""
PraisonAI integration module.

This module provides proper imports for PraisonAI components without requiring
manual sys.path manipulation. It should be used instead of direct imports
from praisonaiagents.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Any, TYPE_CHECKING

logger = logging.getLogger(__name__)

# Add PraisonAI path to sys.path only once during module import
# This is a temporary solution until PraisonAI is properly packaged
_PRAISONAI_PATH_ADDED = False

def _ensure_praisonai_path():
    """Ensure PraisonAI path is available for imports."""
    global _PRAISONAI_PATH_ADDED
    
    if _PRAISONAI_PATH_ADDED:
        return
        
    try:
        # Try to import first to see if it's already available
        import praisonaiagents
        _PRAISONAI_PATH_ADDED = True
        return
    except ImportError:
        pass
    
    # Add path only if needed
    current_dir = Path(__file__).parent.parent.parent
    praisonai_path = current_dir / "PraisonAI" / "src" / "praisonai-agents"
    
    if praisonai_path.exists() and str(praisonai_path) not in sys.path:
        sys.path.insert(0, str(praisonai_path))
        _PRAISONAI_PATH_ADDED = True
        logger.info(f"Added PraisonAI path to sys.path: {praisonai_path}")
    else:
        logger.warning(f"PraisonAI path not found: {praisonai_path}")

# Ensure path is available
_ensure_praisonai_path()

# Import PraisonAI components
try:
    from praisonaiagents import PraisonAIAgents, Agent, Task
    from praisonaiagents.tools import duckduckgo
    
    # Re-export for clean imports
    __all__ = ['PraisonAIAgents', 'Agent', 'Task', 'duckduckgo']
    
except ImportError as e:
    logger.error(f"Failed to import PraisonAI components: {e}")
    # Provide fallback None values to prevent import errors
    PraisonAIAgents = None
    Agent = None  
    Task = None
    duckduckgo = None
    
    __all__ = []


def is_available() -> bool:
    """Check if PraisonAI components are available."""
    return all([PraisonAIAgents, Agent, Task, duckduckgo])


def get_praisonai_version() -> Optional[str]:
    """Get PraisonAI version if available."""
    try:
        import praisonaiagents
        return getattr(praisonaiagents, '__version__', 'unknown')
    except (ImportError, AttributeError):
        return None