"""
Embedding configuration utilities and dimension mappings.

This module centralizes embedding model configurations and provides
utilities for working with different embedding providers and models.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Centralized embedding dimension mappings
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
    "text-embedding-002": 1536,
    "all-mpnet-base-v2": 768,
    "sentence-transformers": 768,
}

# Default fallback dimension
DEFAULT_EMBEDDING_DIMENSION = 1536


def get_embedding_dimensions(model: str) -> int:
    """
    Get embedding dimensions for a given model.
    
    Args:
        model: The embedding model name
        
    Returns:
        The dimension count for the model
    """
    if not model:
        return DEFAULT_EMBEDDING_DIMENSION
    
    model_lower = model.lower().strip()
    
    # Direct lookup first
    if model_lower in EMBEDDING_DIMENSIONS:
        return EMBEDDING_DIMENSIONS[model_lower]
    
    # Partial matching for model names that contain the key
    for key, dimension in EMBEDDING_DIMENSIONS.items():
        if key in model_lower:
            return dimension
    
    logger.warning(f"Unknown embedding model '{model}', using default dimension {DEFAULT_EMBEDDING_DIMENSION}")
    return DEFAULT_EMBEDDING_DIMENSION


def build_embedder_config(provider: str = "openai", model: str = "text-embedding-3-small", **kwargs) -> Dict[str, Any]:
    """
    Build standardized embedder configuration.
    
    Args:
        provider: The embedding provider (e.g., 'openai', 'huggingface')
        model: The embedding model name
        **kwargs: Additional configuration parameters
        
    Returns:
        Standardized embedder configuration dictionary
    """
    config = {
        "model": model,
        "embedding_dims": get_embedding_dimensions(model),
        **kwargs
    }
    
    return {
        "provider": provider,
        "config": config
    }


def validate_embedder_config(embedder_config: Dict[str, Any]) -> bool:
    """
    Validate embedder configuration structure.
    
    Args:
        embedder_config: The embedder configuration to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        if not isinstance(embedder_config, dict):
            return False
        
        if "provider" not in embedder_config:
            return False
        
        if "config" not in embedder_config:
            return False
        
        config = embedder_config["config"]
        if not isinstance(config, dict):
            return False
        
        # Check for required config fields
        if "model" not in config:
            return False
        
        # Validate embedding_dims if present
        if "embedding_dims" in config:
            dims = config["embedding_dims"]
            if not isinstance(dims, int) or dims <= 0:
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating embedder config: {e}")
        return False


def normalize_embedder_config(embedder_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize and enrich embedder configuration.
    
    Args:
        embedder_config: The embedder configuration to normalize
        
    Returns:
        Normalized embedder configuration with missing fields filled in
    """
    if not validate_embedder_config(embedder_config):
        logger.warning("Invalid embedder config provided, using defaults")
        return build_embedder_config()
    
    config = embedder_config.copy()
    
    # Ensure embedding_dims is set
    if "embedding_dims" not in config["config"]:
        model = config["config"]["model"]
        config["config"]["embedding_dims"] = get_embedding_dimensions(model)
        logger.debug(f"Added embedding_dims for model {model}: {config['config']['embedding_dims']}")
    
    return config