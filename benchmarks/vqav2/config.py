# ---------------------------------------------------------------------------
# Model registry — add new providers/models here.
# Keys: provider, api_key_env (env var name), vision (supports image input).
# Distortion and validation always use Mistral (text-only).
# Generation routes to the provider declared here.
# ---------------------------------------------------------------------------
ALLOWED_MODELS = {
    # Mistral — text only (distortion + validation)
    "mistral-large-latest": {
        "provider": "mistral",
        "api_key_env": "MISTRAL_API_KEY",
        "vision": False,
    },
    "mistral-small-latest": {
        "provider": "mistral",
        "api_key_env": "MISTRAL_API_KEY",
        "vision": False,
    },
    # Mistral — vision (Pixtral)
    "pixtral-large-latest": {
        "provider": "mistral",
        "api_key_env": "MISTRAL_API_KEY",
        "vision": True,
    },
    "pixtral-12b-2409": {
        "provider": "mistral",
        "api_key_env": "MISTRAL_API_KEY",
        "vision": True,
    },
    # Gemini — vision
    "gemini-3-flash-preview": {
        "provider": "gemini",
        "api_key_env": "GEMINI_API_KEY",
        "vision": True,
    },
    "gemini-2.5-flash": {
        "provider": "gemini",
        "api_key_env": "GEMINI_API_KEY",
        "vision": True,
    },
}

DEFAULT_MODELS = {
    "distortion": "mistral-large-latest",
    "validation": "mistral-large-latest",
    "generation": "pixtral-12b-2409",
}

DEFAULT_MIU = 0.6
DEFAULT_K_VALUES = [1]
DEFAULT_GENERATION_TEMPERATURE = 0.0
DEFAULT_GENERATION_MAX_TOKENS = 64

MIU_LEVELS = {
    "low": 0.3,
    "medium": 0.6,
    "high": 0.9,
}

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
