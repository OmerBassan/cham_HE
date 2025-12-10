"""
Model backend registry.

Provides functions to register, retrieve, and list available model backends.
"""

from typing import Dict, Type, Optional, List
from chameleon.models.base import ModelBackend
from chameleon.core.schemas import BackendType


# Global registry of backend classes
_BACKEND_REGISTRY: Dict[str, Type[ModelBackend]] = {}


def register_backend(name: str, backend_class: Type[ModelBackend]) -> None:
    """
    Register a model backend.
    
    Args:
        name: Name to register the backend under
        backend_class: Backend class to register
    """
    _BACKEND_REGISTRY[name.lower()] = backend_class


def get_backend(
    backend_type: BackendType,
    model_name: str,
    **kwargs
) -> ModelBackend:
    """
    Get an instance of a model backend.
    
    Args:
        backend_type: Type of backend to use
        model_name: Name of the model to use
        **kwargs: Additional backend configuration
    
    Returns:
        Initialized ModelBackend instance
    """
    backend_name = backend_type.value.lower()
    
    # Lazy import backends to avoid import errors when dependencies aren't installed
    if backend_name not in _BACKEND_REGISTRY:
        _load_backend(backend_name)
    
    if backend_name not in _BACKEND_REGISTRY:
        raise ValueError(f"Unknown backend: {backend_name}. Available: {list_backends()}")
    
    backend_class = _BACKEND_REGISTRY[backend_name]
    return backend_class(model_name=model_name, **kwargs)


def _load_backend(backend_name: str) -> None:
    """Lazy load a backend by name."""
    try:
        if backend_name == "openai":
            from chameleon.models.openai_backend import OpenAIBackend
            register_backend("openai", OpenAIBackend)
        
        elif backend_name == "anthropic":
            from chameleon.models.anthropic_backend import AnthropicBackend
            register_backend("anthropic", AnthropicBackend)
        
        elif backend_name == "mlx":
            from chameleon.models.mlx_backend import MLXBackend
            register_backend("mlx", MLXBackend)
        
        elif backend_name == "cuda_local":
            from chameleon.models.cuda_backend import CUDABackend
            register_backend("cuda_local", CUDABackend)
        
        elif backend_name == "dummy":
            from chameleon.models.dummy_backend import DummyBackend
            register_backend("dummy", DummyBackend)
        
        elif backend_name == "ollama":
            from chameleon.models.ollama_backend import OllamaBackend
            register_backend("ollama", OllamaBackend)
        
        elif backend_name == "huggingface":
            from chameleon.models.huggingface_backend import HuggingFaceBackend
            register_backend("huggingface", HuggingFaceBackend)
    
    except ImportError as e:
        # Backend not available due to missing dependencies
        pass


def list_backends() -> List[str]:
    """
    List all registered backends.
    
    Returns:
        List of backend names
    """
    # Try to load all backends
    for backend_name in ["openai", "anthropic", "mlx", "cuda_local", "dummy", "ollama"]:
        if backend_name not in _BACKEND_REGISTRY:
            _load_backend(backend_name)
    
    return list(_BACKEND_REGISTRY.keys())


def backend_available(backend_type: BackendType) -> bool:
    """
    Check if a backend is available.
    
    Args:
        backend_type: Backend to check
    
    Returns:
        True if backend is available
    """
    try:
        backend = get_backend(backend_type, "test")
        return backend.is_available()
    except (ValueError, ImportError):
        return False


