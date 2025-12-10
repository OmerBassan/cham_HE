"""
Browser helper functions for fetching model information.

These functions are designed to be called from the CLI and use
the browser MCP tools to fetch real-time model information.
"""

from typing import List, Dict, Optional, Tuple
from chameleon.cli.model_lookup import (
    VENDOR_MODEL_DOCS,
    FALLBACK_MODELS,
    get_search_url,
    get_huggingface_search_url,
)


def get_vendor_models_url(vendor: str) -> str:
    """Get the URL to fetch models for a vendor."""
    return VENDOR_MODEL_DOCS.get(vendor.lower(), "")


def get_fallback_models_for_vendor(vendor: str) -> List[str]:
    """Get fallback models when browser lookup is not available."""
    return FALLBACK_MODELS.get(vendor.lower(), ["custom"])


def parse_openai_models(snapshot_text: str) -> List[str]:
    """Parse OpenAI models from snapshot."""
    import re
    models = []
    
    # Look for model IDs in the snapshot (Updated Dec 2025)
    patterns = [
        # GPT-5 series
        r'(gpt-5\.1(?:-\w+)?)',
        r'(gpt-5(?:-mini|-nano|-pro)?)',
        # GPT-4.1 series
        r'(gpt-4\.1(?:-mini|-nano)?)',
        # GPT-4o series
        r'(gpt-4o(?:-mini)?(?:-\d{4}-\d{2}-\d{2})?)',
        r'(gpt-4-turbo(?:-\d{4}-\d{2}-\d{2})?)',
        # Reasoning models
        r'(o3)',
        r'(o4-mini)',
        r'(o1(?:-pro|-preview|-mini)?)',
        # Legacy
        r'(gpt-3\.5-turbo(?:-\d{4})?)',
        # Open models
        r'(gpt-os-\d+b)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, snapshot_text, re.IGNORECASE)
        models.extend(matches)
    
    # Deduplicate and clean
    seen = set()
    clean_models = []
    for m in models:
        m_lower = m.lower()
        if m_lower not in seen:
            seen.add(m_lower)
            clean_models.append(m)
    
    return clean_models if clean_models else FALLBACK_MODELS["openai"]


def parse_anthropic_models(snapshot_text: str) -> List[str]:
    """Parse Anthropic models from snapshot."""
    import re
    models = []
    
    # Updated Dec 2025 for Claude 4.5 series
    patterns = [
        # Claude 4.5 series (Latest)
        r'(claude-sonnet-4-5(?:-\d{8})?)',
        r'(claude-haiku-4-5(?:-\d{8})?)',
        r'(claude-opus-4-5(?:-\d{8})?)',
        r'(claude-opus-4-1(?:-\d{8})?)',
        # Claude 3.5 series
        r'(claude-3-5-sonnet-\d{8})',
        r'(claude-3-5-haiku-\d{8})',
        r'(claude-3-opus-\d{8})',
        r'(claude-3-sonnet-\d{8})',
        r'(claude-3-haiku-\d{8})',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, snapshot_text, re.IGNORECASE)
        models.extend(matches)
    
    seen = set()
    clean_models = []
    for m in models:
        m_lower = m.lower()
        if m_lower not in seen:
            seen.add(m_lower)
            clean_models.append(m)
    
    return clean_models if clean_models else FALLBACK_MODELS["anthropic"]


def parse_google_models(snapshot_text: str) -> List[str]:
    """Parse Google/Gemini models from snapshot."""
    import re
    models = []
    
    patterns = [
        r'(gemini-2\.0-flash(?:-exp)?)',
        r'(gemini-1\.5-pro(?:-\d+)?)',
        r'(gemini-1\.5-flash(?:-8b)?(?:-\d+)?)',
        r'(gemini-1\.0-pro)',
        r'(gemini-pro)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, snapshot_text, re.IGNORECASE)
        models.extend(matches)
    
    seen = set()
    clean_models = []
    for m in models:
        m_lower = m.lower()
        if m_lower not in seen:
            seen.add(m_lower)
            clean_models.append(m)
    
    return clean_models if clean_models else FALLBACK_MODELS["google"]


def parse_mistral_models(snapshot_text: str) -> List[str]:
    """Parse Mistral models from snapshot."""
    import re
    models = []
    
    patterns = [
        r'(mistral-large(?:-latest|-\d+)?)',
        r'(mistral-medium(?:-latest)?)',
        r'(mistral-small(?:-latest|-\d+)?)',
        r'(open-mistral-7b)',
        r'(open-mixtral-8x7b)',
        r'(open-mixtral-8x22b)',
        r'(codestral(?:-latest)?)',
        r'(pixtral-\d+b(?:-\d+)?)',
        r'(ministral-\d+b(?:-\d+)?)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, snapshot_text, re.IGNORECASE)
        models.extend(matches)
    
    seen = set()
    clean_models = []
    for m in models:
        m_lower = m.lower()
        if m_lower not in seen:
            seen.add(m_lower)
            clean_models.append(m)
    
    return clean_models if clean_models else FALLBACK_MODELS["mistral"]


def parse_models_for_vendor(vendor: str, snapshot_text: str) -> List[str]:
    """Parse models from snapshot based on vendor."""
    parsers = {
        "openai": parse_openai_models,
        "anthropic": parse_anthropic_models,
        "google": parse_google_models,
        "mistral": parse_mistral_models,
    }
    
    parser = parsers.get(vendor.lower())
    if parser:
        return parser(snapshot_text)
    
    return FALLBACK_MODELS.get(vendor.lower(), ["custom"])


def format_model_search_results(search_results: str, query: str) -> List[Dict[str, str]]:
    """
    Parse search results for model information.
    
    Returns list of potential model matches with name and description.
    """
    import re
    
    results = []
    
    # Look for model names in search results
    # Common patterns for model names
    patterns = [
        r'([a-zA-Z]+-\d+(?:\.\d+)?[a-zA-Z-]*)',  # e.g., gpt-4, claude-3
        r'([A-Z][a-zA-Z]+-\d+[A-Z]?(?:-[a-zA-Z]+)?)',  # e.g., Llama-3B, Mistral-7B
    ]
    
    matches = []
    for pattern in patterns:
        found = re.findall(pattern, search_results)
        matches.extend(found)
    
    # Deduplicate
    seen = set()
    for match in matches:
        if match.lower() not in seen and len(match) > 3:
            seen.add(match.lower())
            results.append({
                "name": match,
                "description": f"Found in search for '{query}'"
            })
    
    return results[:10]  # Limit to 10 results


# Instructions for browser-based model lookup (to be printed in CLI)
BROWSER_LOOKUP_INSTRUCTIONS = """
🌐 Fetching latest models from {vendor} documentation...
   URL: {url}
"""

BROWSER_SEARCH_INSTRUCTIONS = """
🔍 Searching for model: {query}
   This will search online for model information.
"""

