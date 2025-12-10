"""
Model lookup utilities using web browser to fetch latest model information.

Uses browser MCP tools to:
- Fetch current model lists from vendor documentation
- Validate model names against API docs
- Search for model information when user enters unknown models
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    vendor: str
    description: str = ""
    context_length: Optional[int] = None
    pricing: Optional[str] = None
    capabilities: List[str] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []


# URLs for model documentation by vendor
VENDOR_MODEL_DOCS = {
    "openai": "https://platform.openai.com/docs/models",
    "anthropic": "https://docs.anthropic.com/en/docs/about-claude/models",
    "google": "https://ai.google.dev/gemini-api/docs/models/gemini",
    "mistral": "https://docs.mistral.ai/getting-started/models/",
    "cohere": "https://docs.cohere.com/docs/models",
    "meta": "https://llama.meta.com/docs/model-cards-and-prompt-formats/",
}

# Fallback model lists (used when browser lookup fails)
# Updated: Dec 2025 from vendor documentation
FALLBACK_MODELS = {
    "openai": [
        # GPT-5 series (Latest)
        "gpt-5.1",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5-pro",
        # GPT-4.1 series
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        # Reasoning models
        "o3",
        "o4-mini",
        # GPT-4o series (Still supported)
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        # Legacy
        "gpt-3.5-turbo",
    ],
    "anthropic": [
        # Claude 4.5 series (Latest)
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "claude-opus-4-5",
        "claude-opus-4-1",
        # With specific dates
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-5-20251101",
        # Claude 3.5 series (Legacy)
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
    ],
    "google": [
        "gemini-2.0-flash-exp",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.0-pro",
    ],
    "mistral": [
        "mistral-large-latest",
        "mistral-medium-latest",
        "mistral-small-latest",
        "open-mistral-7b",
        "open-mixtral-8x7b",
        "open-mixtral-8x22b",
        "codestral-latest",
        "pixtral-12b",
        "ministral-8b",
    ],
    "cohere": [
        "command-r-plus",
        "command-r",
        "command",
        "command-light",
    ],
    "ollama": [
        "llama3.2",
        "llama3.1",
        "mistral",
        "codellama",
        "phi3",
        "gemma2",
        "qwen2.5",
    ],
    "huggingface": [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "microsoft/Phi-3-mini-4k-instruct",
        "google/gemma-2-9b-it",
        "Qwen/Qwen2.5-7B-Instruct",
    ],
}


class ModelLookup:
    """
    Handles model lookup using browser tools.
    
    This class provides methods to fetch and validate model information
    from vendor documentation using browser automation.
    """
    
    def __init__(self, use_browser: bool = True):
        """
        Initialize ModelLookup.
        
        Args:
            use_browser: Whether to use browser for lookups (requires MCP tools)
        """
        self.use_browser = use_browser
        self._cache: Dict[str, List[str]] = {}
    
    def get_vendor_url(self, vendor: str) -> Optional[str]:
        """Get the documentation URL for a vendor."""
        return VENDOR_MODEL_DOCS.get(vendor.lower())
    
    def get_fallback_models(self, vendor: str) -> List[str]:
        """Get fallback model list for a vendor."""
        return FALLBACK_MODELS.get(vendor.lower(), [])
    
    def parse_models_from_snapshot(self, snapshot_text: str, vendor: str) -> List[str]:
        """
        Parse model names from a browser snapshot.
        
        Args:
            snapshot_text: Text content from browser snapshot
            vendor: Vendor name for context
        
        Returns:
            List of model names found
        """
        models = []
        
        # Vendor-specific parsing patterns
        patterns = {
            "openai": [
                r'gpt-4o(?:-mini)?',
                r'gpt-4(?:-turbo)?(?:-\d{4}-\d{2}-\d{2})?',
                r'gpt-3\.5-turbo(?:-\d{4})?',
                r'o1(?:-preview|-mini)?',
                r'text-embedding-\d+-\w+',
            ],
            "anthropic": [
                r'claude-3(?:\.\d)?-(?:opus|sonnet|haiku)(?:-\d{8})?',
                r'claude-2(?:\.\d)?',
            ],
            "google": [
                r'gemini-(?:2\.0|1\.5|1\.0)-(?:pro|flash|ultra)(?:-\w+)?',
            ],
            "mistral": [
                r'mistral-(?:large|medium|small)(?:-latest)?',
                r'open-mistral-\w+',
                r'open-mixtral-\w+',
                r'codestral(?:-latest)?',
            ],
        }
        
        vendor_patterns = patterns.get(vendor.lower(), [])
        
        for pattern in vendor_patterns:
            matches = re.findall(pattern, snapshot_text, re.IGNORECASE)
            models.extend(matches)
        
        # Remove duplicates and sort
        return sorted(list(set(models)))
    
    def format_model_list(self, models: List[str], max_display: int = 8) -> str:
        """Format model list for display."""
        if not models:
            return "No models found"
        
        if len(models) <= max_display:
            return ", ".join(models)
        
        displayed = models[:max_display]
        remaining = len(models) - max_display
        return f"{', '.join(displayed)} (+{remaining} more)"


# Singleton instance
_model_lookup = None

def get_model_lookup() -> ModelLookup:
    """Get or create the ModelLookup singleton."""
    global _model_lookup
    if _model_lookup is None:
        _model_lookup = ModelLookup()
    return _model_lookup


async def fetch_models_from_browser(vendor: str) -> Tuple[List[str], bool]:
    """
    Fetch models from vendor documentation using browser.
    
    This is an async function that should be called from the CLI
    using browser MCP tools.
    
    Args:
        vendor: Vendor name
    
    Returns:
        Tuple of (model_list, success_bool)
    """
    lookup = get_model_lookup()
    
    # Check cache first
    if vendor in lookup._cache:
        return lookup._cache[vendor], True
    
    # Return fallback - actual browser calls happen in CLI
    return lookup.get_fallback_models(vendor), False


def get_search_url(query: str) -> str:
    """Get a search URL for model information."""
    encoded_query = query.replace(" ", "+")
    return f"https://www.google.com/search?q={encoded_query}+model+card+api"


def get_huggingface_search_url(model_name: str) -> str:
    """Get HuggingFace search URL for a model."""
    encoded = model_name.replace(" ", "+").replace("/", "%2F")
    return f"https://huggingface.co/models?search={encoded}"

