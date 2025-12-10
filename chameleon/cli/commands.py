"""
CLI commands for the Chameleon framework.

Provides the main CLI interface for:
- Creating and managing projects
- Running evaluations
- Analyzing results
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

from chameleon.core.schemas import Modality, BackendType
from chameleon.core.project import Project, list_projects
from chameleon.core.config import ChameleonConfig
from chameleon.cli.validators import (
    get_validated_input,
    get_validated_input_with_default,
    validate_project_name,
    validate_miu_values,
    validate_model_name,
    validate_vendor,
    validate_api_key,
    validate_integer,
    get_yes_no,
    get_choice,
    get_choice_with_skip,
    confirm_input,
    ValidationError,
)
from chameleon.cli.browser_helpers import (
    get_vendor_models_url,
    get_fallback_models_for_vendor,
    parse_models_for_vendor,
    BROWSER_LOOKUP_INSTRUCTIONS,
)
from chameleon.cli.model_lookup import get_search_url, get_huggingface_search_url
from chameleon.cli.data_upload import run_file_upload_flow, validate_uploaded_data

# Default projects directory (capital P)
DEFAULT_PROJECTS_DIR = "Projects"

# Flag to enable browser lookups (can be disabled for testing)
ENABLE_BROWSER_LOOKUPS = True


def fetch_models_with_browser(vendor: str) -> Tuple[List[str], bool]:
    """
    Fetch latest models from vendor documentation using browser.
    
    This function uses browser MCP tools if available.
    Falls back to hardcoded lists if browser is not available.
    
    Args:
        vendor: Vendor name (openai, anthropic, etc.)
    
    Returns:
        Tuple of (model_list, fetched_from_browser)
    """
    if not ENABLE_BROWSER_LOOKUPS:
        return get_fallback_models_for_vendor(vendor), False
    
    url = get_vendor_models_url(vendor)
    if not url:
        return get_fallback_models_for_vendor(vendor), False
    
    print(BROWSER_LOOKUP_INSTRUCTIONS.format(vendor=vendor.title(), url=url))
    
    # The actual browser call happens through MCP - we return the URL
    # and signal that browser lookup should be attempted
    return get_fallback_models_for_vendor(vendor), False


def search_model_info(query: str) -> str:
    """
    Get search URL for unknown model.
    
    Returns URL that can be used with browser tools.
    """
    return get_search_url(query)


def print_banner():
    """Print the Chameleon banner."""
    # Enable ANSI colors on Windows
    import sys
    if sys.platform == 'win32':
        import os
        os.system('')  # Enable ANSI escape sequences on Windows
    
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    
    banner = f"""
{GREEN}     ██████╗██╗  ██╗ █████╗ ███╗   ███╗███████╗██╗     ███████╗ ██████╗ ███╗   ██╗
    ██╔════╝██║  ██║██╔══██╗████╗ ████║██╔════╝██║     ██╔════╝██╔═══██╗████╗  ██║
    ██║     ███████║███████║██╔████╔██║█████╗  ██║     █████╗  ██║   ██║██╔██╗ ██║
    ██║     ██╔══██║██╔══██║██║╚██╔╝██║██╔══╝  ██║     ██╔══╝  ██║   ██║██║╚██╗██║
    ╚██████╗██║  ██║██║  ██║██║ ╚═╝ ██║███████╗███████╗███████╗╚██████╔╝██║ ╚████║
     ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝{RESET}

{CYAN}    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                        LLM Robustness Benchmark:                          ║
    ║        Evaluate model performance under lexical & LLM distortion          ║
    ╚═══════════════════════════════════════════════════════════════════════════╝{RESET}
    """
    print(banner)


def cmd_init(args):
    """Initialize a new project interactively with full validation."""
    print_banner()
    print("📁 Creating a new evaluation project\n")
    print("   (Enter 'q' at any prompt to quit)\n")
    
    projects_dir = Path(args.projects_dir)
    
    try:
        # ===== 1. PROJECT NAME =====
        print("─" * 50)
        print("📝 STEP 1: Project Name")
        print("─" * 50)
        
        if args.name:
            project_name = args.name
            print(f"   Using provided name: {project_name}")
        else:
            while True:
                project_name = get_validated_input(
                    "\n   Enter project name: ",
                    validate_project_name,
                    "Invalid project name",
                    confirm=True
                )
                
                # Check if project exists
                project_path = projects_dir / project_name
                if project_path.exists():
                    print(f"   ❌ Project '{project_name}' already exists at {project_path}")
                    if not get_yes_no("   Try a different name?", default=True):
                        return 1
                else:
                    break
        
        # ===== 2. MODALITY =====
        print("\n" + "─" * 50)
        print("📊 STEP 2: Input Modality")
        print("─" * 50)
        
        if args.modality:
            modality = Modality(args.modality)
            print(f"   Using provided modality: {modality.value}")
        else:
            modality_choices = [m.value for m in Modality]
            modality_str = get_choice(
                "\n   Select input modality:",
                modality_choices,
                default="text"
            )
            modality = Modality(modality_str)
            
            confirmed, _ = confirm_input("Modality", modality.value)
            if not confirmed:
                modality_str = get_choice("\n   Select input modality:", modality_choices)
                modality = Modality(modality_str)
        
        # ===== 3. VENDOR & MODEL =====
        print("\n" + "─" * 50)
        print("🤖 STEP 3: Target Model Configuration")
        print("─" * 50)
        
        # Model suggestions by vendor (updated Dec 2025)
        model_suggestions = {
            "openai": ["gpt-5.1", "gpt-5-mini", "gpt-4.1", "gpt-4o", "gpt-4o-mini", "o3", "o4-mini"],
            "anthropic": ["claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-5", "claude-opus-4-1"],
            "google": ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"],
            "mistral": ["mistral-large-latest", "mistral-small-latest", "codestral-latest", "open-mistral-7b"],
            "local": ["mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Llama-3.1-8B-Instruct", "custom"],
            "ollama": ["llama3.2", "mistral", "codellama", "phi3", "custom"],
        }
        
        # Get vendor
        if args.backend:
            vendor = args.backend
            print(f"   Using provided vendor: {vendor}")
        else:
            vendor_choices = ["openai", "anthropic", "google", "mistral", "local", "ollama"]
            vendor = get_choice_with_skip(
                "\n   Select model vendor (or press Enter for default):",
                vendor_choices,
                default="openai",
                allow_skip=True,
                confirm=True
            )
        
        # Get model name - with live API lookup
        target_api_key = None  # Will store key for later use (defined here for scope)
        
        if args.model:
            model_name = args.model
            print(f"   Using provided model: {model_name}")
        else:
            suggestions = model_suggestions.get(vendor, ["custom"])
            
            # Try to fetch latest models via API
            fetched_models = None
            existing_key = os.getenv(f"{vendor.upper()}_API_KEY")
            
            # If no key in environment, ask user for one to fetch models
            if not existing_key and vendor in ["openai", "mistral"]:
                print(f"\n   💡 To fetch the latest {vendor} models, we need an API key.")
                has_key = get_yes_no(f"   Do you have a {vendor.upper()} API key?", default=True)
                
                if has_key:
                    key_input = input(f"   Enter {vendor.upper()}_API_KEY: ").strip()
                    if key_input:
                        existing_key = key_input
                        target_api_key = key_input  # Save for later
                        print(f"   ✓ Key received")
            
            if existing_key and vendor in ["openai", "mistral"]:
                print(f"\n   🔄 Fetching latest {vendor} models from API...")
                try:
                    if vendor == "openai":
                        import openai
                        client = openai.OpenAI(api_key=existing_key)
                        models_response = client.models.list()
                        
                        # Filter to CHAT models only (exclude moderation, embedding, whisper, tts, dall-e)
                        excluded = ['moderation', 'embedding', 'whisper', 'tts', 'dall-e', 'davinci', 'babbage', 'curie', 'ada']
                        chat_models = []
                        for m in models_response.data:
                            model_id = m.id.lower()
                            # Include GPT and o-series reasoning models
                            if ('gpt-' in model_id or model_id.startswith('o1') or model_id.startswith('o3') or model_id.startswith('o4')):
                                # Exclude non-chat variants
                                if not any(ex in model_id for ex in excluded):
                                    chat_models.append(m.id)
                        
                        # Smart sort: gpt-5 > gpt-4 > o-series, newest first
                        def model_sort_key(name):
                            n = name.lower()
                            if 'gpt-5' in n: return (0, name)
                            if 'gpt-4o' in n and 'mini' not in n: return (1, name)
                            if 'gpt-4o-mini' in n: return (2, name)
                            if 'gpt-4' in n: return (3, name)
                            if n.startswith('o4'): return (4, name)
                            if n.startswith('o3'): return (5, name)
                            if n.startswith('o1'): return (6, name)
                            if 'gpt-3' in n: return (7, name)
                            return (8, name)
                        
                        fetched_models = sorted(chat_models, key=model_sort_key)[:15]
                        
                    elif vendor == "mistral":
                        from mistralai import Mistral
                        client = Mistral(api_key=existing_key)
                        models_response = client.models.list()
                        # Filter to main models (exclude embeddings)
                        chat_models = [m.id for m in models_response.data if 'embed' not in m.id.lower()]
                        fetched_models = sorted(chat_models)[:15]
                    
                    if fetched_models:
                        print(f"   ✅ Found {len(fetched_models)} chat models from {vendor} API (live)")
                        suggestions = fetched_models
                except Exception as e:
                    print(f"   ⚠️ Could not fetch models: {e}")
                    print(f"   📋 Using known models list")
            elif vendor not in ["openai", "mistral"]:
                # For other vendors, offer to check docs
                print(f"\n   Would you like to check the latest {vendor} models from their docs?")
                use_browser = get_yes_no("   Open docs URL?", default=False)
                
                if use_browser:
                    url = get_vendor_models_url(vendor)
                    if url:
                        print(f"\n   🌐 Check: {url}")
                        print(f"   📋 Using known models list below:")
            
            # Show models in a nice list
            print(f"\n   Available {vendor} models:")
            for i, model in enumerate(suggestions[:8], 1):
                default_mark = " ← default" if i == 1 else ""
                print(f"      {i}. {model}{default_mark}")
            
            if len(suggestions) > 8:
                print(f"      ... and {len(suggestions) - 8} more")
            
            print(f"\n   Press Enter for default: {suggestions[0]}")
            print(f"   Or type a custom model name")
            
            model_name = get_validated_input_with_default(
                f"\n   Enter model name or number: ",
                lambda x: suggestions[int(x)-1] if x.isdigit() and 1 <= int(x) <= len(suggestions) else validate_model_name(x),
                suggestions[0],
                "Invalid model name",
                confirm=True
            )
            
            # If model not in known list, try fuzzy matching
            if model_name not in suggestions and model_name != suggestions[0]:
                # Fuzzy match: normalize and find closest
                def normalize(s):
                    return s.lower().replace('-', '').replace('_', '').replace('.', '').replace(' ', '')
                
                input_norm = normalize(model_name)
                best_match = None
                best_score = 0
                
                for s in suggestions:
                    s_norm = normalize(s)
                    # Check if input is substring or very similar
                    if input_norm in s_norm or s_norm in input_norm:
                        score = len(set(input_norm) & set(s_norm)) / max(len(input_norm), len(s_norm))
                        if score > best_score:
                            best_score = score
                            best_match = s
                    # Also check character overlap
                    overlap = len(set(input_norm) & set(s_norm)) / max(len(input_norm), len(s_norm))
                    if overlap > 0.7 and overlap > best_score:
                        best_score = overlap
                        best_match = s
                
                if best_match and best_score > 0.5:
                    print(f"\n   🔍 '{model_name}' not found. Did you mean '{best_match}'?")
                    use_suggestion = get_yes_no(f"   Use '{best_match}' instead?", default=True)
                    if use_suggestion:
                        model_name = best_match
                        print(f"   ✓ Using: {model_name}")
                    else:
                        print(f"\n   ℹ️ Keeping '{model_name}' as entered.")
                else:
                    print(f"\n   ℹ️ '{model_name}' is not in our known models list.")
                    search_for_it = get_yes_no("   Would you like to search for this model online?", default=False)
                    
                    if search_for_it:
                        search_url = get_search_url(f"{vendor} {model_name} API model")
                        print(f"\n   🔍 Search URL: {search_url}")
                        print(f"   📄 HuggingFace: {get_huggingface_search_url(model_name)}")
                        
                        keep_model = get_yes_no(f"\n   Continue with '{model_name}'?", default=True)
                        if not keep_model:
                            model_name = get_validated_input(
                                "\n   Enter correct model name: ",
                                validate_model_name,
                                "Invalid model name",
                                confirm=True
                            )
        
        # Map vendor to backend type
        vendor_to_backend = {
            "openai": BackendType.OPENAI,
            "anthropic": BackendType.ANTHROPIC,
            "google": BackendType.GOOGLE,
            "mistral": BackendType.MISTRAL,
            "local": BackendType.CUDA_LOCAL,
            "ollama": BackendType.OLLAMA,
        }
        backend_type = vendor_to_backend.get(vendor, BackendType.OPENAI)
        
        # ===== 4. DISTORTION SETTINGS =====
        print("\n" + "─" * 50)
        print("🔧 STEP 4: Distortion Settings")
        print("─" * 50)
        
        # Miu values
        print("\n   Miu (μ) values control distortion intensity (0.0 = none, 1.0 = max)")
        print("   Format: comma-separated values (0.1, 0.3, 0.5)")
        print("           or range with step (0.0-0.9:0.1)")
        print("   Press Enter for default: 0.0-0.9:0.1")
        
        default_miu = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        if hasattr(args, 'miu') and args.miu:
            miu_values = validate_miu_values(args.miu)
        else:
            miu_values = get_validated_input_with_default(
                "\n   Enter miu values: ",
                lambda x: validate_miu_values(x),
                default_miu,
                "Invalid miu format",
                confirm=True
            )
        
        print(f"   ✓ Miu values: {miu_values}")
        
        # Distortions per question
        print("\n   How many distortion variants per question per miu level?")
        print("   Press Enter for default: 10")
        
        if hasattr(args, 'distortions_per_question') and args.distortions_per_question:
            distortions_per_question = args.distortions_per_question
        else:
            distortions_per_question = get_validated_input_with_default(
                "   Enter distortions per question: ",
                lambda x: validate_integer(x, min_val=1, max_val=50),
                10,
                "Invalid number (must be 1-50)",
                confirm=True
            )
        
        print(f"   ✓ Distortions per question: {distortions_per_question}")
        
        # ===== 4b. DISTORTION ENGINE =====
        print("\n" + "─" * 50)
        print("⚙️ STEP 4b: Distortion Engine Configuration")
        print("─" * 50)
        
        print("\n   Choose how to run the distortion generation engine:")
        engine_type_choices = ["api", "local", "ollama"]
        engine_type = get_choice_with_skip(
            "\n   Distortion engine type:",
            engine_type_choices,
            default="api",
            confirm=True
        )
        
        # Configure based on engine type
        distortion_engine_config = {
            "engine_type": engine_type,
        }
        
        if engine_type == "api":
            print("\n   Using cloud API for distortion generation")
            api_vendor_choices = ["mistral", "openai", "anthropic", "google"]
            dist_vendor = get_choice_with_skip(
                "\n   Select API vendor for distortion:",
                api_vendor_choices,
                default="mistral",
                confirm=True
            )
            
            dist_model_suggestions = {
                "mistral": ["mistral-large-latest", "mistral-small-latest", "codestral-latest", "open-mistral-nemo"],
                "openai": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                "anthropic": ["claude-3-5-haiku-latest", "claude-3-haiku-20240307"],
                "google": ["gemini-1.5-flash", "gemini-1.5-pro"],
            }
            
            suggestions = dist_model_suggestions.get(dist_vendor, ["custom"])
            
            print(f"\n   Available {dist_vendor.title()} models for distortion:")
            for i, model in enumerate(suggestions, 1):
                default_marker = " ← default" if i == 1 else ""
                print(f"      {i}. {model}{default_marker}")
            
            print(f"\n   Enter number (1-{len(suggestions)}), model name, or press Enter for default")
            
            model_input = input(f"   Model: ").strip()
            
            if not model_input:
                dist_model = suggestions[0]
            elif model_input.isdigit() and 1 <= int(model_input) <= len(suggestions):
                dist_model = suggestions[int(model_input) - 1]
            else:
                dist_model = model_input
            
            print(f"   ✓ Selected: {dist_model}")
            
            distortion_engine_config.update({
                "vendor": dist_vendor,
                "model_name": dist_model,
                "api_key_env_var": f"{dist_vendor.upper()}_API_KEY",
            })
            
        elif engine_type == "local":
            print("\n   Using local model for distortion generation")
            print("   This uses HuggingFace transformers with GPU acceleration")
            
            local_model_choices = [
                "mistralai/Mistral-7B-Instruct-v0.2",
                "meta-llama/Llama-3-8B-Instruct",
                "microsoft/Phi-3-mini-4k-instruct",
                "custom"
            ]
            
            dist_model = get_choice_with_skip(
                "\n   Select local model:",
                local_model_choices,
                default="mistralai/Mistral-7B-Instruct-v0.2",
                confirm=True
            )
            
            if dist_model == "custom":
                dist_model = get_validated_input(
                    "\n   Enter HuggingFace model ID or local path: ",
                    validate_model_name,
                    "Invalid model path",
                    confirm=True
                )
            
            # Quantization option
            print("\n   Quantization reduces memory usage (recommended for large models)")
            quant_choices = ["none", "4bit", "8bit"]
            quantization = get_choice_with_skip(
                "\n   Select quantization:",
                quant_choices,
                default="4bit",
                confirm=False  # Don't need to confirm technical settings
            )
            
            distortion_engine_config.update({
                "vendor": "huggingface",
                "model_name": dist_model,
                "quantization": None if quantization == "none" else quantization,
                "use_gpu": True,
                "device": "auto",
            })
            
        elif engine_type == "ollama":
            print("\n   Using Ollama for local inference")
            print("   Make sure Ollama is running: ollama serve")
            
            ollama_models = ["mistral", "llama3", "codellama", "phi3", "custom"]
            dist_model = get_choice_with_skip(
                "\n   Select Ollama model:",
                ollama_models,
                default="mistral",
                confirm=True
            )
            
            if dist_model == "custom":
                dist_model = get_validated_input(
                    "\n   Enter Ollama model name: ",
                    validate_model_name,
                    "Invalid model name",
                    confirm=True
                )
            
            distortion_engine_config.update({
                "vendor": "ollama",
                "model_name": dist_model,
                "api_base_url": "http://localhost:11434",
            })
        
        print(f"\n   ✓ Distortion engine: {engine_type} / {distortion_engine_config.get('model_name', 'default')}")
        
        # ===== 5. DISTORTION ENGINE API KEY (if API mode) =====
        distortion_api_key = None
        if engine_type == "api":
            dist_vendor = distortion_engine_config.get("vendor", "mistral")
            dist_env_var = f"{dist_vendor.upper()}_API_KEY"
            
            print("\n" + "─" * 50)
            print("🔑 STEP 5a: Distortion Engine API Key")
            print("─" * 50)
            
            existing_dist_key = os.getenv(dist_env_var)
            
            if existing_dist_key:
                print(f"\n   Found existing {dist_env_var} in environment.")
                use_existing = get_yes_no(f"   Use existing key?", default=True)
                if use_existing:
                    distortion_api_key = existing_dist_key
                    print("   ✓ Using existing API key")
            
            if not distortion_api_key:
                print(f"\n   Enter your {dist_vendor.title()} API key for distortion generation")
                print("   (This will be stored in the project's .env file)")
                print("   (Press Enter to skip - you can add it later)")
                
                try:
                    raw_key = input(f"   {dist_env_var}: ").strip()
                    if raw_key:
                        distortion_api_key = raw_key
                        print("   ✓ API key saved")
                    else:
                        print("   ⏭️ Skipping - add it later to .env")
                except Exception as e:
                    print(f"   ⚠️ {e} - you can add it later to .env file")
        
        # ===== 5b. TARGET MODEL API KEY =====
        print("\n" + "─" * 50)
        print("🔑 STEP 5b: Target Model API Key")
        print("─" * 50)
        
        api_key = None
        env_var_name = f"{vendor.upper()}_API_KEY"
        existing_key = os.getenv(env_var_name)
        
        # Check if we already collected this key when fetching models
        if target_api_key:
            print(f"\n   ✓ Using API key from model fetch step")
            api_key = target_api_key
        # If distortion vendor same as target vendor and we already have key
        elif distortion_api_key and distortion_engine_config.get("vendor") == vendor:
            print(f"\n   Same vendor as distortion engine ({vendor})")
            api_key = distortion_api_key
            print("   ✓ Using same API key")
        elif existing_key:
            print(f"\n   Found existing {env_var_name} in environment.")
            use_existing = get_yes_no(f"   Use existing key?", default=True)
            if use_existing:
                api_key = existing_key
                print("   ✓ Using existing API key")
        
        if not api_key and vendor not in ["local", "ollama", "dummy"]:
            print(f"\n   Enter your {vendor.title()} API key for model evaluation")
            print("   (This will be stored in the project's .env file)")
            print("   (Press Enter to skip - you can add it later to .env file)")
            
            try:
                raw_key = input(f"   {env_var_name}: ").strip()
                if raw_key:
                    api_key = raw_key
                    print("   ✓ API key saved")
                else:
                    print("   ⏭️ Skipping API key - add it later to .env")
            except Exception as e:
                print(f"   ⚠️ {e} - you can add it later to .env file")
        
        # ===== 6. DESCRIPTION =====
        print("\n" + "─" * 50)
        print("📝 STEP 6: Project Description (optional)")
        print("─" * 50)
        
        description = args.description or input("\n   Enter description (or press Enter to skip): ").strip()
        
        # ===== SUMMARY & CONFIRMATION =====
        # Format distortion engine info for display
        dist_engine_display = f"{distortion_engine_config.get('engine_type', 'local')} / {distortion_engine_config.get('model_name', 'mistral-7b')}"
        
        print("\n" + "═" * 50)
        print("📋 PROJECT CONFIGURATION SUMMARY")
        print("═" * 50)
        print(f"""
   Project Name:          {project_name}
   Modality:              {modality.value}
   
   TARGET MODEL:
   Vendor:                {vendor}
   Model:                 {model_name}
   Backend:               {backend_type.value}
   
   DISTORTION SETTINGS:
   Miu Values:            {miu_values}
   Distortions/Question:  {distortions_per_question}
   Distortion Engine:     {dist_engine_display}
   
   API Key:               {'✓ Configured' if api_key else '✗ Not set'}
   Description:           {description or '(none)'}
        """)
        
        # Allow user to modify settings before creation
        while True:
            response = get_yes_no("\n   Create project with these settings?", default=True)
            
            if response:
                break  # User confirmed, proceed with creation
            
            # User said no - ask what to change
            print("\n   What would you like to change?")
            change_options = [
                "project_name",
                "modality", 
                "vendor",
                "model",
                "miu_values",
                "distortions_per_question",
                "distortion_engine",
                "api_key",
                "description",
                "cancel"
            ]
            
            for i, opt in enumerate(change_options, 1):
                display_name = opt.replace("_", " ").title()
                print(f"   {i}. {display_name}")
            
            choice = input("\n   Enter number or name: ").strip().lower()
            
            # Parse choice
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(change_options):
                    choice = change_options[choice_idx]
            except ValueError:
                # Try matching by name
                choice = choice.replace(" ", "_")
            
            if choice == "cancel":
                print("\n   ❌ Project creation cancelled.")
                return 1
            
            elif choice == "project_name":
                while True:
                    project_name = get_validated_input(
                        "\n   Enter new project name: ",
                        validate_project_name,
                        "Invalid project name",
                        confirm=True
                    )
                    project_path = projects_dir / project_name
                    if project_path.exists():
                        print(f"   ❌ Project '{project_name}' already exists")
                    else:
                        break
            
            elif choice == "modality":
                modality_choices = [m.value for m in Modality]
                modality_str = get_choice("\n   Select input modality:", modality_choices)
                modality = Modality(modality_str)
            
            elif choice == "vendor":
                vendor_choices = ["openai", "anthropic", "google", "mistral", "local", "ollama"]
                vendor = get_choice("\n   Select model vendor:", vendor_choices)
                
                # Update backend type
                vendor_to_backend = {
                    "openai": BackendType.OPENAI,
                    "anthropic": BackendType.ANTHROPIC,
                    "google": BackendType.OPENAI,
                    "mistral": BackendType.OPENAI,
                    "local": BackendType.CUDA_LOCAL,
                    "ollama": BackendType.OLLAMA,
                }
                backend_type = vendor_to_backend.get(vendor, BackendType.OPENAI)
                
                # Update env var name
                env_var_name = f"{vendor.upper()}_API_KEY"
            
            elif choice == "model":
                model_suggestions = {
                    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1-preview"],
                    "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
                    "google": ["gemini-1.5-pro", "gemini-1.5-flash"],
                    "mistral": ["mistral-large-latest", "mistral-medium"],
                    "local": ["mistral-7b-instruct-v0.3", "llama-3.1-8b", "phi-3-mini"],
                    "ollama": ["llama3", "mistral", "codellama"],
                }
                suggestions = model_suggestions.get(vendor, ["custom"])
                print(f"\n   Common {vendor} models: {', '.join(suggestions[:4])}")
                
                model_name = get_validated_input(
                    f"\n   Enter model name: ",
                    validate_model_name,
                    "Invalid model name",
                    confirm=True
                )
            
            elif choice == "miu_values":
                print("\n   Format: comma-separated (0.1, 0.3, 0.5) or range (0.0-0.9:0.1)")
                miu_values = get_validated_input(
                    "   Enter miu values: ",
                    validate_miu_values,
                    "Invalid miu format",
                    confirm=True
                )
            
            elif choice == "distortions_per_question":
                distortions_per_question = get_validated_input(
                    "\n   Enter distortions per question: ",
                    lambda x: validate_integer(x, min_val=1, max_val=50),
                    "Invalid number",
                    confirm=True
                )
            
            elif choice == "api_key":
                env_var_name = f"{vendor.upper()}_API_KEY"
                print(f"\n   Enter your {vendor.title()} API key")
                try:
                    raw_key = input(f"   {env_var_name}: ").strip()
                    if raw_key:
                        api_key = validate_api_key(raw_key)
                    else:
                        api_key = None
                        print("   ⏭️ API key cleared")
                except ValidationError as e:
                    print(f"   ⚠️ {e}")
            
            elif choice == "distortion_engine":
                print("\n   Reconfigure distortion engine:")
                engine_type_choices = ["api", "local", "ollama"]
                engine_type = get_choice_with_skip(
                    "\n   Distortion engine type:",
                    engine_type_choices,
                    default="api",
                    confirm=True
                )
                
                distortion_engine_config = {"engine_type": engine_type}
                
                if engine_type == "api":
                    api_vendor_choices = ["mistral", "openai", "anthropic", "google"]
                    dist_vendor = get_choice_with_skip(
                        "\n   Select API vendor:",
                        api_vendor_choices,
                        default="mistral",
                        confirm=True
                    )
                    dist_model = get_validated_input(
                        "\n   Enter model name: ",
                        validate_model_name,
                        "Invalid model name",
                        confirm=True
                    )
                    distortion_engine_config.update({
                        "vendor": dist_vendor,
                        "model_name": dist_model,
                        "api_key_env_var": f"{dist_vendor.upper()}_API_KEY",
                    })
                elif engine_type == "local":
                    dist_model = get_validated_input(
                        "\n   Enter HuggingFace model ID: ",
                        validate_model_name,
                        "Invalid model ID",
                        confirm=True
                    )
                    distortion_engine_config.update({
                        "vendor": "huggingface",
                        "model_name": dist_model,
                        "use_gpu": True,
                    })
                elif engine_type == "ollama":
                    dist_model = get_validated_input(
                        "\n   Enter Ollama model name: ",
                        validate_model_name,
                        "Invalid model name",
                        confirm=True
                    )
                    distortion_engine_config.update({
                        "vendor": "ollama",
                        "model_name": dist_model,
                        "api_base_url": "http://localhost:11434",
                    })
            
            elif choice == "description":
                description = input("\n   Enter new description: ").strip()
            
            else:
                print(f"   ❌ Unknown option: {choice}")
                continue
            
            # Show updated summary
            dist_engine_display = f"{distortion_engine_config.get('engine_type', 'local')} / {distortion_engine_config.get('model_name', 'mistral-7b')}"
            
            print("\n" + "═" * 50)
            print("📋 UPDATED CONFIGURATION")
            print("═" * 50)
            print(f"""
   Project Name:          {project_name}
   Modality:              {modality.value}
   
   TARGET MODEL:
   Vendor:                {vendor}
   Model:                 {model_name}
   Backend:               {backend_type.value}
   
   DISTORTION SETTINGS:
   Miu Values:            {miu_values}
   Distortions/Question:  {distortions_per_question}
   Distortion Engine:     {dist_engine_display}
   
   API Key:               {'✓ Configured' if api_key else '✗ Not set'}
   Description:           {description or '(none)'}
            """)
        
        # ===== CREATE PROJECT =====
        print(f"\n🚀 Creating project '{project_name}'...")
        
        project = Project.create(
            name=project_name,
            modality=modality,
            model_name=model_name,
            backend_type=backend_type,
            base_dir=projects_dir,
            description=description,
            distortion_levels=miu_values,
            distortions_per_question=distortions_per_question,
        )
        
        # Create project config.yaml with all settings
        project_config = {
            "project": {
                "name": project_name,
                "description": description,
                "modality": modality.value,
            },
            "target_model": {
                "vendor": vendor,
                "name": model_name,
                "backend": backend_type.value,
            },
            "distortion": {
                "miu_values": miu_values,
                "distortions_per_question": distortions_per_question,
                "engine": distortion_engine_config,
            },
            "data": {
                "original_data_path": "original_data",
                "distorted_data_path": "distorted_data",
                "results_path": "results",
                "analysis_path": "analysis",
            },
        }
        
        import yaml
        config_yaml_path = project.project_dir / "config.yaml"
        with open(config_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(project_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        # Create .env file with API keys
        env_path = project.project_dir / ".env"
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(f"# API Keys for {project_name}\n\n")
            
            # Distortion engine API key
            dist_vendor = distortion_engine_config.get("vendor", "")
            if distortion_engine_config.get("engine_type") == "api" and dist_vendor:
                dist_env_var = f"{dist_vendor.upper()}_API_KEY"
                f.write(f"# Distortion Engine ({dist_vendor})\n")
                f.write(f"{dist_env_var}={distortion_api_key or ''}\n\n")
            
            # Target model API key (if different vendor)
            if vendor not in ["local", "ollama", "dummy"]:
                # Don't duplicate if same vendor as distortion engine
                if vendor != dist_vendor or distortion_engine_config.get("engine_type") != "api":
                    f.write(f"# Target Model ({vendor})\n")
                    f.write(f"{env_var_name}={api_key or ''}\n")
        
        keys_configured = []
        if distortion_api_key:
            keys_configured.append(f"{dist_vendor.upper()}_API_KEY")
        if api_key and (vendor != dist_vendor or distortion_engine_config.get("engine_type") != "api"):
            keys_configured.append(env_var_name)
        
        if keys_configured:
            print(f"   ✓ Created .env file with: {', '.join(keys_configured)}")
        else:
            print(f"   ✓ Created .env file (add API keys manually)")
        
        # Create .gitignore to protect .env
        gitignore_path = project.project_dir / ".gitignore"
        with open(gitignore_path, 'w', encoding='utf-8') as f:
            f.write("# Ignore sensitive files\n")
            f.write(".env\n")
            f.write("*.log\n")
            f.write("__pycache__/\n")
        
        print(f"\n✅ Project created successfully!")
        print(f"\n📁 Project structure:")
        print(f"   {project.project_dir}/")
        print(f"   ├── original_data/    # Place your original data here")
        print(f"   ├── distorted_data/   # Distorted versions will be stored here")
        print(f"   ├── results/          # Model outputs from evaluations")
        print(f"   ├── analysis/         # Analysis outputs and reports")
        print(f"   ├── config.yaml       # Project configuration")
        print(f"   ├── .env              # API keys (git-ignored)")
        print(f"   ├── .gitignore")
        print(f"   ├── project_config.yaml")
        print(f"   └── README.md")
        
        # ===== 7. FILE UPLOAD =====
        print(f"\n🎯 Project created! Now let's add your data.")
        
        upload_now = get_yes_no("\n   Would you like to upload your data files now?", default=True)
        
        if upload_now:
            files_uploaded = run_file_upload_flow(project.project_dir)
            
            if files_uploaded:
                print(f"\n✅ Data uploaded successfully!")
                
                # Ask if user wants to start distortion generation
                start_distortion = get_yes_no(
                    "\n   Would you like to generate distortions now?",
                    default=True
                )
                
                if start_distortion:
                    print(f"\n🔄 Starting distortion generation...")
                    print(f"   Run: python cli.py distort --project {project_name}")
                    # TODO: Call cmd_distort directly here
                    return 0
        
        print(f"\n📋 Next steps:")
        print(f"   1. Add your data to: {project.original_data_dir}")
        print(f"   2. Generate distortions:  python cli.py distort --project {project_name}")
        print(f"   3. Run evaluation:        python cli.py evaluate --project {project_name}")
        print(f"   4. Analyze results:       python cli.py analyze --project {project_name}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n   ❌ Project creation cancelled by user.")
        return 1
    except ValidationError as e:
        print(f"\n   ❌ Validation error: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Failed to create project: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_list(args):
    """List all projects."""
    print_banner()
    print("📁 Available Projects\n")
    
    projects_dir = Path(args.projects_dir)
    projects = list_projects(projects_dir)
    
    if not projects:
        print("   No projects found.")
        print(f"\n💡 Create a new project with: python cli.py init")
        return 0
    
    for project in projects:
        status_icon = "✅" if project.get("exists") else "❌"
        name = project.get("name", "Unknown")
        
        if "error" in project:
            print(f"   {status_icon} {name} (Error: {project['error']})")
        else:
            config = project.get("config", {})
            project_path = projects_dir / name
            
            # Target model info (check both formats for backwards compatibility)
            target_model = config.get("target_model", {})
            target_vendor = target_model.get("vendor", config.get("backend_type", "unknown"))
            target_name = target_model.get("name", config.get("model_name", "unknown"))
            
            # Distortion model info (check both distortion_config and distortion keys)
            distortion = config.get("distortion_config", config.get("distortion", {}))
            distortion_engine = distortion.get("engine", {})
            distortion_model = distortion_engine.get("model_name", "mistral-large-latest")
            distortion_vendor = distortion_engine.get("vendor", "mistral")
            miu_values = distortion.get("miu_values", [])
            dpq = distortion.get("distortions_per_question", 10)
            
            # Format miu values nicely
            if miu_values:
                miu_str = ", ".join([str(m) for m in miu_values])
            else:
                miu_str = "N/A"
            
            print(f"\n{'='*70}")
            print(f"   {status_icon} {name}")
            print(f"{'='*70}")
            
            # ── Configuration Metadata ──
            print(f"\n   ┌─ Configuration")
            print(f"   │  📂 Path: {project_path}")
            print(f"   │  📝 Description: {config.get('description', 'N/A')}")
            print(f"   │  🎨 Modality: {config.get('modality', 'text')}")
            print(f"   │")
            print(f"   │  🎯 Target Model")
            print(f"   │     • Vendor: {target_vendor}")
            print(f"   │     • Model: {target_name}")
            print(f"   │")
            engine_type = distortion_engine.get('engine_type', 'api')
            print(f"   │  🔀 Distortion Engine")
            print(f"   │     • Type: {engine_type.upper()}")
            print(f"   │     • Model: {distortion_vendor}/{distortion_model}")
            if engine_type == "local":
                # Only show local-specific settings for local engine
                print(f"   │     • Max Workers: {distortion_engine.get('max_workers', 4)}")
                print(f"   │     • Batch Size: {distortion_engine.get('batch_size', 8)}")
                if distortion_engine.get('model_path'):
                    print(f"   │     • Model Path: {distortion_engine.get('model_path')}")
            print(f"   │")
            print(f"   │  📊 Distortion Settings")
            print(f"   │     • Miu Values: [{miu_str}]")
            print(f"   │     • Distortions/Question: {dpq}")
            print(f"   │")
            
            # Metadata
            metadata = config.get('metadata', {})
            if metadata:
                print(f"   │  🕐 Metadata")
                print(f"   │     • Created: {metadata.get('created_at', 'N/A')}")
                print(f"   │     • Updated: {metadata.get('updated_at', 'N/A')}")
                print(f"   │     • Version: {metadata.get('version', 'N/A')}")
                print(f"   │")
            
            print(f"   └─────────────────────────────────")
            
            # ── Full Directory Tree ──
            print(f"\n   ┌─ Directory Structure")
            _print_tree(project_path, prefix="   │  ")
            print(f"   └─────────────────────────────────")
            
            print()
    
    return 0


def _print_tree(directory: Path, prefix: str = "", is_last: bool = True, max_depth: int = 4, current_depth: int = 0):
    """Print directory tree recursively."""
    if current_depth > max_depth:
        return
    
    if not directory.exists():
        return
    
    # Get all items, excluding hidden files and __pycache__
    items = sorted([
        item for item in directory.iterdir() 
        if not item.name.startswith('.') and item.name != '__pycache__'
    ], key=lambda x: (x.is_file(), x.name.lower()))
    
    for i, item in enumerate(items):
        is_last_item = (i == len(items) - 1)
        connector = "└── " if is_last_item else "├── "
        
        if item.is_file():
            # Get file size
            size = item.stat().st_size
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size / (1024 * 1024):.1f} MB"
            
            print(f"{prefix}{connector}📄 {item.name} ({size_str})")
        else:
            # Count items in subdirectory
            try:
                subitem_count = len(list(item.iterdir()))
            except PermissionError:
                subitem_count = "?"
            
            print(f"{prefix}{connector}📂 {item.name}/ ({subitem_count} items)")
            
            # Recurse into subdirectory
            new_prefix = prefix + ("    " if is_last_item else "│   ")
            _print_tree(item, new_prefix, is_last_item, max_depth, current_depth + 1)


def cmd_status(args):
    """Show status of a specific project."""
    print_banner()
    
    project_name = args.project
    projects_dir = Path(args.projects_dir)
    project_path = projects_dir / project_name
    
    if not project_path.exists():
        print(f"❌ Project '{project_name}' not found")
        return 1
    
    try:
        project = Project.load(project_path)
        status = project.get_status()
        
        print(f"📊 Project Status: {project_name}\n")
        
        config = status.get("config", {})
        print(f"Configuration:")
        print(f"   Modality:  {config.get('modality', 'N/A')}")
        print(f"   Model:     {config.get('model_name', 'N/A')}")
        print(f"   Backend:   {config.get('backend_type', 'N/A')}")
        print(f"   Created:   {config.get('metadata', {}).get('created_at', 'N/A')}")
        
        # Load project-specific config.yaml if exists
        config_yaml_path = project_path / "config.yaml"
        if config_yaml_path.exists():
            import yaml
            with open(config_yaml_path, 'r', encoding='utf-8') as f:
                proj_config = yaml.safe_load(f)
            
            distortion = proj_config.get("distortion", {})
            if distortion:
                print(f"\nDistortion Settings:")
                print(f"   Miu Values:            {distortion.get('miu_values', 'N/A')}")
                print(f"   Distortions/Question:  {distortion.get('distortions_per_question', 'N/A')}")
                print(f"   Engine:                {distortion.get('engine', 'N/A')}")
        
        # Check for .env file
        env_path = project_path / ".env"
        print(f"\nAPI Key Status:")
        if env_path.exists():
            print(f"   ✓ .env file present")
        else:
            print(f"   ✗ No .env file")
        
        print(f"\nFiles:")
        files = status.get("files", {})
        for dir_name, counts in files.items():
            if counts:
                total = sum(counts.values())
                details = ", ".join([f"{ext}: {count}" for ext, count in counts.items()])
                print(f"   {dir_name}/: {total} files ({details})")
            else:
                print(f"   {dir_name}/: empty")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error loading project: {e}")
        return 1


def cmd_analyze(args):
    """Run comprehensive analysis on project results."""
    print_banner()
    
    project_name = args.project
    projects_dir = Path(args.projects_dir)
    project_path = projects_dir / project_name
    
    if not project_path.exists():
        print(f"❌ Project '{project_name}' not found")
        return 1
    
    try:
        from chameleon.analysis import run_analysis
        
        result = run_analysis(project_name, str(projects_dir))
        
        return 0 if result.get("status") == "complete" else 1
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Install with: pip install matplotlib seaborn scipy statsmodels")
        return 1
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_distort(args):
    """Generate distortions for project data using the unified runner."""
    print_banner()
    
    project_name = args.project
    projects_dir = Path(args.projects_dir)
    project_path = projects_dir / project_name
    
    if not project_path.exists():
        print(f"❌ Project '{project_name}' not found")
        return 1
    
    try:
        # Check for data
        original_data_dir = project_path / "original_data"
        data_files = list(original_data_dir.glob("*.csv"))
        
        if not data_files:
            print(f"\n❌ No data files found in {original_data_dir}")
            print(f"   Run file upload first or copy CSV files manually.")
            return 1
        
        print(f"\n📁 Data files found: {len(data_files)}")
        for f in data_files:
            print(f"   • {f.name}")
        
        # Confirm
        if not args.yes:
            proceed = get_yes_no("\n   Start distortion generation?", default=True)
            if not proceed:
                print("   Cancelled.")
                return 0
        
        # Use the unified runner
        from chameleon.distortion.runner import run_distortions
        
        result = run_distortions(project_name, str(projects_dir))
        
        print(f"\n🎯 Next steps:")
        print(f"   1. Review distorted data in: {project_path / 'distorted_data'}")
        print(f"   2. Run evaluation: python cli.py evaluate --project {project_name}")
        print(f"   3. Analyze results: python cli.py analyze --project {project_name}")
        
        return 0 if result.get("status") == "complete" else 1
        
    except Exception as e:
        print(f"\n❌ Error during distortion generation: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_evaluate(args):
    """Evaluate target model on distorted questions."""
    import yaml
    from dotenv import load_dotenv, set_key
    
    print_banner()
    
    project_name = args.project
    projects_dir = Path(args.projects_dir)
    project_path = projects_dir / project_name
    
    if not project_path.exists():
        print(f"❌ Project '{project_name}' not found")
        return 1
    
    # Load project config to get target model vendor
    config_path = project_path / "config.yaml"
    env_path = project_path / ".env"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    target_vendor = config.get("target_model", {}).get("vendor", "openai").upper()
    target_model = config.get("target_model", {}).get("name", "gpt-4o")
    api_key_name = f"{target_vendor}_API_KEY"
    
    def get_and_validate_key(env_path, api_key_name, target_vendor, target_model):
        """Get API key and validate it works."""
        import os
        load_dotenv(env_path, override=True)
        api_key = os.getenv(api_key_name)
        
        if not api_key:
            print(f"⚠️  {api_key_name} not found in project .env file")
            print(f"   Target model: {target_vendor.lower()} / {target_model}")
            print()
            return None, False
        
        # Validate the key by making a test request
        if target_vendor == "OPENAI":
            try:
                import openai
                client = openai.OpenAI(api_key=api_key)
                # Quick validation - list models (minimal API call)
                client.models.list()
                return api_key, True
            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg or "invalid_api_key" in error_msg:
                    print(f"❌ Invalid {api_key_name} in .env file")
                    print(f"   Error: API key is incorrect or expired")
                    return api_key, False
                elif "429" in error_msg:
                    # Rate limited but key is valid
                    return api_key, True
                else:
                    print(f"⚠️  Could not validate API key: {e}")
                    return api_key, True  # Assume valid, let it fail later if not
        
        return api_key, True  # Non-OpenAI vendors, assume valid
    
    import os
    api_key, is_valid = get_and_validate_key(env_path, api_key_name, target_vendor, target_model)
    
    # If no key or invalid key, prompt for new one
    while not api_key or not is_valid:
        print()
        new_key = input(f"Enter your {target_vendor} API key (or 'q' to cancel): ").strip()
        
        if new_key.lower() == 'q':
            print("❌ Evaluation cancelled.")
            return 1
        
        if not new_key:
            print("❌ No API key provided.")
            continue
        
        # Save to .env file
        if not env_path.exists():
            env_path.touch()
        
        set_key(str(env_path), api_key_name, new_key)
        os.environ[api_key_name] = new_key
        print(f"✅ {api_key_name} saved to {env_path}")
        
        # Validate the new key
        api_key, is_valid = get_and_validate_key(env_path, api_key_name, target_vendor, target_model)
        
        if is_valid:
            print("✅ API key validated successfully!")
            break
    
    print()
    print(f"🎯 Running evaluation for project: {project_name}")
    print(f"   Target model: {target_vendor.lower()} / {target_model}")
    print()
    
    try:
        from chameleon.evaluation import run_evaluation
        
        result = run_evaluation(project_name, str(projects_dir))
        
        return 0 if result.get("status") == "complete" else 1
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Install with: pip install openai")
        return 1
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_delete(args):
    """Delete a project with double confirmation."""
    import shutil
    
    print_banner()
    
    project_name = args.project
    projects_dir = Path(args.projects_dir)
    project_path = projects_dir / project_name
    
    if not project_path.exists():
        print(f"❌ Project '{project_name}' not found")
        return 1
    
    # Show project info
    print(f"🗑️  DELETE PROJECT: {project_name}")
    print(f"   Location: {project_path}")
    print()
    
    # Count files
    file_count = sum(1 for _ in project_path.rglob("*") if _.is_file())
    print(f"   ⚠️  This will permanently delete {file_count} files!")
    print()
    
    if not args.force:
        # First confirmation
        confirm1 = input(f"   Are you sure you want to delete '{project_name}'? (yes/no): ").strip().lower()
        
        if confirm1 != 'yes':
            print("   ❌ Deletion cancelled.")
            return 0
        
        # Second confirmation - type project name
        print()
        confirm2 = input(f"   Type the project name to confirm: ").strip()
        
        if confirm2 != project_name:
            print(f"   ❌ Names don't match. Deletion cancelled.")
            return 0
    
    # Delete the project
    try:
        shutil.rmtree(project_path)
        print(f"\n   ✅ Project '{project_name}' deleted successfully.")
        return 0
    except Exception as e:
        print(f"\n   ❌ Error deleting project: {e}")
        return 1


def cmd_edit(args):
    """Edit project configuration interactively."""
    import yaml
    
    print_banner()
    
    project_name = args.project
    projects_dir = Path(args.projects_dir)
    project_path = projects_dir / project_name
    config_path = project_path / "config.yaml"
    
    if not project_path.exists():
        print(f"❌ Project '{project_name}' not found")
        return 1
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return 1
    
    # Load current config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"✏️  EDIT PROJECT: {project_name}")
    print("=" * 50)
    print("\nCurrent configuration:")
    print("-" * 30)
    
    # Show current settings
    target_model = config.get("target_model", {})
    distortion = config.get("distortion", {})
    
    print(f"   Target Vendor: {target_model.get('vendor', 'N/A')}")
    print(f"   Target Model: {target_model.get('name', 'N/A')}")
    print(f"   Miu Values: {distortion.get('miu_values', 'N/A')}")
    print(f"   Distortions/Question: {distortion.get('distortions_per_question', 'N/A')}")
    print(f"   Distortion Engine: {distortion.get('engine_type', 'N/A')}")
    print(f"   Distortion Model: {distortion.get('model', 'N/A')}")
    print()
    
    changes_made = False
    
    # Edit options
    print("What would you like to edit?")
    print("   1. Target model (vendor/name)")
    print("   2. Miu values")
    print("   3. Distortions per question")
    print("   4. Distortion engine settings")
    print("   5. All settings (go through everything)")
    print("   q. Cancel")
    print()
    
    choice = input("Enter choice (1-5 or q): ").strip().lower()
    
    if choice == 'q':
        print("   ❌ Edit cancelled.")
        return 0
    
    if choice in ['1', '5']:
        print("\n--- Target Model ---")
        vendors = ["openai", "anthropic", "mistral"]
        current_vendor = target_model.get('vendor', 'openai')
        print(f"   Current vendor: {current_vendor}")
        new_vendor = input(f"   New vendor ({'/'.join(vendors)}) [Enter to keep]: ").strip().lower()
        if new_vendor and new_vendor in vendors:
            config['target_model']['vendor'] = new_vendor
            changes_made = True
        
        current_model = target_model.get('name', '')
        print(f"   Current model: {current_model}")
        new_model = input(f"   New model name [Enter to keep]: ").strip()
        if new_model:
            config['target_model']['name'] = new_model
            changes_made = True
    
    if choice in ['2', '5']:
        print("\n--- Miu Values ---")
        current_miu = distortion.get('miu_values', [])
        print(f"   Current: {current_miu}")
        print("   Enter new values as comma-separated (e.g., 0.0,0.1,0.5,0.9)")
        print("   Or range (e.g., 0.0-0.9:0.1)")
        new_miu = input("   New miu values [Enter to keep]: ").strip()
        if new_miu:
            # Parse miu values
            if '-' in new_miu and ':' in new_miu:
                # Range format
                try:
                    range_part, step = new_miu.split(':')
                    start, end = range_part.split('-')
                    start, end, step = float(start), float(end), float(step)
                    miu_list = []
                    current = start
                    while current <= end + 0.001:
                        miu_list.append(round(current, 1))
                        current += step
                    config['distortion']['miu_values'] = miu_list
                    changes_made = True
                except:
                    print("   ⚠️ Invalid format, keeping current values")
            else:
                # Comma-separated
                try:
                    miu_list = [float(x.strip()) for x in new_miu.split(',')]
                    config['distortion']['miu_values'] = miu_list
                    changes_made = True
                except:
                    print("   ⚠️ Invalid format, keeping current values")
    
    if choice in ['3', '5']:
        print("\n--- Distortions per Question ---")
        current_dpq = distortion.get('distortions_per_question', 10)
        print(f"   Current: {current_dpq}")
        new_dpq = input("   New value [Enter to keep]: ").strip()
        if new_dpq:
            try:
                config['distortion']['distortions_per_question'] = int(new_dpq)
                changes_made = True
            except:
                print("   ⚠️ Invalid number, keeping current value")
    
    if choice in ['4', '5']:
        print("\n--- Distortion Engine ---")
        current_engine = distortion.get('engine_type', 'api')
        print(f"   Current engine: {current_engine}")
        new_engine = input("   New engine (api/local) [Enter to keep]: ").strip().lower()
        if new_engine in ['api', 'local']:
            config['distortion']['engine_type'] = new_engine
            changes_made = True
        
        current_dm = distortion.get('model', '')
        print(f"   Current distortion model: {current_dm}")
        new_dm = input("   New model [Enter to keep]: ").strip()
        if new_dm:
            config['distortion']['model'] = new_dm
            changes_made = True
    
    if not changes_made:
        print("\n   ℹ️ No changes made.")
        return 0
    
    # Save config
    print("\n--- Summary of Changes ---")
    print(f"   Target Model: {config['target_model'].get('vendor')}/{config['target_model'].get('name')}")
    print(f"   Miu Values: {config['distortion'].get('miu_values')}")
    print(f"   Distortions/Question: {config['distortion'].get('distortions_per_question')}")
    print()
    
    save = input("Save changes? (y/n): ").strip().lower()
    if save != 'y':
        print("   ❌ Changes discarded.")
        return 0
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"   ✅ Configuration saved to {config_path}")
    
    # Ask if they want to regenerate distortions
    print()
    regen = input("   Do you want to regenerate distortions with new settings? (y/n): ").strip().lower()
    if regen == 'y':
        print("\n   🔄 Starting distortion regeneration...")
        # Create args-like object for cmd_distort
        class DistortArgs:
            project = project_name
            projects_dir = str(projects_dir)
            yes = False
        
        return cmd_distort(DistortArgs())
    
    return 0


def cmd_workflow(args):
    """Run the complete Chameleon workflow."""
    print_banner()
    
    project_name = args.project
    projects_dir = Path(args.projects_dir)
    project_path = projects_dir / project_name
    
    if not project_path.exists():
        print(f"❌ Project '{project_name}' not found")
        return 1
    
    try:
        from chameleon.workflow import run_workflow
        
        result = run_workflow(
            project_name,
            str(projects_dir),
            skip_distortion=args.skip_distortion,
            skip_evaluation=args.skip_evaluation,
            skip_analysis=args.skip_analysis,
        )
        
        return 0 if result.get("status") == "success" else 1
        
    except Exception as e:
        print(f"\n❌ Error during workflow: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_config(args):
    """Show or update global configuration."""
    print_banner()
    
    config = ChameleonConfig.load()
    
    if args.show:
        print("🔧 Global Configuration\n")
        
        for key, value in config.to_dict().items():
            print(f"   {key}: {value}")
        
        print("\n📋 Validation:")
        validation = config.validate()
        for key, value in validation.items():
            status = "✅" if value else "❌"
            print(f"   {status} {key}")
        
        return 0
    
    # Save config
    config.save()
    print("✅ Configuration saved")
    return 0


def cmd_help(args):
    """Show help information."""
    print_banner()
    print("""
📋 Available Commands:

   init       Create a new evaluation project (interactive)
   list       List all projects
   status     Show status of a project
   distort    Generate distortions for project data
   evaluate   Run model evaluation on distorted data
   analyze    Run analysis on project results
   workflow   Run complete end-to-end workflow
   config     Show/update global configuration
   help       Show this help message

🚀 Quick Start:

   1. Create a project:
      python cli.py init

   2. Add your data to the project (or use the upload prompt)

   3. Generate distortions:
      python cli.py distort --project my_project

   4. Evaluate target model:
      python cli.py evaluate --project my_project

   5. Analyze results:
      python cli.py analyze --project my_project

🎯 Or run everything at once:
      python cli.py workflow --project my_project

📚 For more information, see the README.md file.
    """)
    return 0


def main(argv: Optional[List[str]] = None):
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Chameleon - LLM Robustness Testing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--projects-dir",
        default=DEFAULT_PROJECTS_DIR,
        help=f"Base directory for projects (default: {DEFAULT_PROJECTS_DIR})"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # init command
    init_parser = subparsers.add_parser("init", help="Create a new project")
    init_parser.add_argument("--name", "-n", help="Project name")
    init_parser.add_argument("--modality", "-m", choices=[m.value for m in Modality], help="Input modality")
    init_parser.add_argument("--model", help="Model name")
    init_parser.add_argument("--backend", "-b", choices=[b.value for b in BackendType], help="Backend type")
    init_parser.add_argument("--description", "-d", help="Project description")
    init_parser.add_argument("--miu", help="Miu values (e.g., '0.0-0.9:0.1' or '0.1,0.3,0.5')")
    init_parser.add_argument("--distortions-per-question", type=int, help="Distortions per question")
    init_parser.set_defaults(func=cmd_init)
    
    # list command
    list_parser = subparsers.add_parser("list", help="List all projects")
    list_parser.set_defaults(func=cmd_list)
    
    # status command
    status_parser = subparsers.add_parser("status", help="Show project status")
    status_parser.add_argument("--project", "-p", required=True, help="Project name")
    status_parser.set_defaults(func=cmd_status)
    
    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run analysis")
    analyze_parser.add_argument("--project", "-p", required=True, help="Project name")
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # distort command
    distort_parser = subparsers.add_parser("distort", help="Generate distortions")
    distort_parser.add_argument("--project", "-p", required=True, help="Project name")
    distort_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    distort_parser.set_defaults(func=cmd_distort)
    
    # evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate target model")
    evaluate_parser.add_argument("--project", "-p", required=True, help="Project name")
    evaluate_parser.set_defaults(func=cmd_evaluate)
    
    # workflow command
    workflow_parser = subparsers.add_parser("workflow", help="Run complete workflow")
    workflow_parser.add_argument("--project", "-p", required=True, help="Project name")
    workflow_parser.add_argument("--skip-distortion", action="store_true", help="Skip distortion stage")
    workflow_parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation stage")
    workflow_parser.add_argument("--skip-analysis", action="store_true", help="Skip analysis stage")
    workflow_parser.set_defaults(func=cmd_workflow)
    
    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a project")
    delete_parser.add_argument("--project", "-p", required=True, help="Project name")
    delete_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation")
    delete_parser.set_defaults(func=cmd_delete)
    
    # edit command
    edit_parser = subparsers.add_parser("edit", help="Edit project configuration")
    edit_parser.add_argument("--project", "-p", required=True, help="Project name")
    edit_parser.set_defaults(func=cmd_edit)
    
    # config command
    config_parser = subparsers.add_parser("config", help="Global configuration")
    config_parser.add_argument("--show", "-s", action="store_true", help="Show configuration")
    config_parser.set_defaults(func=cmd_config)
    
    # help command
    help_parser = subparsers.add_parser("help", help="Show help")
    help_parser.set_defaults(func=cmd_help)
    
    # Parse arguments
    args = parser.parse_args(argv)
    
    if args.command is None:
        cmd_help(args)
        return 0
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
