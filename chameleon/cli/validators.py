"""
Input validation and confirmation utilities for CLI.

Provides:
- Input validation functions
- Confirmation prompts
- Type-safe input collection
"""

import re
from typing import Optional, List, Tuple, Any, Callable
from pathlib import Path


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


def confirm_input(prompt: str, value: Any, allow_change: bool = True) -> Tuple[bool, Any]:
    """
    Ask user to confirm an input value.
    
    Args:
        prompt: Description of what was entered
        value: The value to confirm
        allow_change: Allow user to re-enter if not confirmed
    
    Returns:
        Tuple of (confirmed, final_value)
    """
    print(f"\n   You entered: {value}")
    
    while True:
        response = input("   Is this correct? (y/n/q): ").strip().lower()
        
        if response in ['y', 'yes']:
            return True, value
        elif response in ['n', 'no']:
            if allow_change:
                return False, None
            else:
                print("   Value cannot be changed. Please confirm or quit.")
        elif response in ['q', 'quit']:
            raise KeyboardInterrupt("User cancelled")
        else:
            print("   Please enter 'y' for yes, 'n' for no, or 'q' to quit.")


def get_validated_input(
    prompt: str,
    validator: Callable[[str], Any],
    error_message: str = "Invalid input",
    max_attempts: int = 3,
    confirm: bool = True
) -> Any:
    """
    Get validated input from user with optional confirmation.
    
    Args:
        prompt: Input prompt to display
        validator: Function that validates and transforms input (raises ValidationError if invalid)
        error_message: Message to show on validation failure
        max_attempts: Maximum validation attempts before failing
        confirm: Whether to ask for confirmation
    
    Returns:
        Validated and confirmed value
    """
    attempts = 0
    
    while attempts < max_attempts:
        try:
            raw_input = input(prompt).strip()
            
            if not raw_input:
                print(f"   ❌ Input cannot be empty")
                attempts += 1
                continue
            
            # Validate and transform
            value = validator(raw_input)
            
            # Confirm if requested
            if confirm:
                confirmed, final_value = confirm_input(prompt.rstrip(": "), value)
                if confirmed:
                    return final_value
                else:
                    print("   Let's try again...")
                    continue
            else:
                return value
                
        except ValidationError as e:
            print(f"   ❌ {error_message}: {e}")
            attempts += 1
        except ValueError as e:
            print(f"   ❌ {error_message}: {e}")
            attempts += 1
    
    raise ValidationError(f"Maximum attempts ({max_attempts}) exceeded")


def get_validated_input_with_default(
    prompt: str,
    validator: Callable[[str], Any],
    default: Any,
    error_message: str = "Invalid input",
    max_attempts: int = 10,
    confirm: bool = True
) -> Any:
    """
    Get validated input with a default value if user presses Enter.
    
    Args:
        prompt: Input prompt to display
        validator: Function that validates and transforms input
        default: Default value to use if user presses Enter
        error_message: Message to show on validation failure
        max_attempts: Maximum validation attempts before failing
        confirm: Whether to ask for confirmation
    
    Returns:
        Validated and confirmed value, or default
    """
    attempts = 0
    
    while attempts < max_attempts:
        try:
            raw_input = input(prompt).strip()
            
            # If empty, use default
            if not raw_input:
                print(f"   → Using default: {default}")
                if confirm:
                    confirmed, _ = confirm_input("Default value", default)
                    if confirmed:
                        return default
                    else:
                        print("   Let's try again...")
                        continue
                else:
                    return default
            
            # Validate and transform
            value = validator(raw_input)
            
            # Confirm if requested
            if confirm:
                confirmed, final_value = confirm_input(prompt.rstrip(": "), value)
                if confirmed:
                    return final_value
                else:
                    print("   Let's try again...")
                    continue
            else:
                return value
                
        except ValidationError as e:
            print(f"   ❌ {error_message}: {e}")
            attempts += 1
        except ValueError as e:
            print(f"   ❌ {error_message}: {e}")
            attempts += 1
    
    raise ValidationError(f"Maximum attempts ({max_attempts}) exceeded")


def validate_project_name(name: str) -> str:
    """Validate project name (alphanumeric, underscores, hyphens)."""
    if not name:
        raise ValidationError("Project name cannot be empty")
    
    # Allow alphanumeric, underscores, hyphens
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', name):
        raise ValidationError(
            "Project name must start with a letter and contain only "
            "letters, numbers, underscores, and hyphens"
        )
    
    if len(name) > 50:
        raise ValidationError("Project name must be 50 characters or less")
    
    return name


def validate_miu_values(miu_input: str) -> List[float]:
    """
    Validate and parse miu values.
    
    Accepts:
    - Comma-separated values: "0.1, 0.3, 0.5"
    - Range with step: "0.0-0.9:0.1" (start-end:step)
    - Mixed: "0.0, 0.5, 0.8-1.0:0.1"
    
    Returns:
        Sorted list of unique miu values
    """
    miu_values = set()
    
    parts = [p.strip() for p in miu_input.split(',')]
    
    for part in parts:
        if ':' in part and '-' in part:
            # Range format: start-end:step
            try:
                range_part, step_str = part.split(':')
                start_str, end_str = range_part.split('-')
                
                start = float(start_str)
                end = float(end_str)
                step = float(step_str)
                
                if step <= 0:
                    raise ValidationError(f"Step must be positive: {step}")
                if start > end:
                    raise ValidationError(f"Start ({start}) must be <= end ({end})")
                
                current = start
                while current <= end + 0.0001:  # Small epsilon for float comparison
                    if 0.0 <= current <= 1.0:
                        miu_values.add(round(current, 2))
                    current += step
                    
            except ValueError:
                raise ValidationError(f"Invalid range format: {part}. Use 'start-end:step'")
        else:
            # Single value
            try:
                value = float(part)
                if not 0.0 <= value <= 1.0:
                    raise ValidationError(f"Miu value must be between 0.0 and 1.0: {value}")
                miu_values.add(round(value, 2))
            except ValueError:
                raise ValidationError(f"Invalid miu value: {part}")
    
    if not miu_values:
        raise ValidationError("No valid miu values provided")
    
    return sorted(list(miu_values))


def validate_model_name(model_name: str) -> str:
    """Validate model name (non-empty string)."""
    if not model_name:
        raise ValidationError("Model name cannot be empty")
    
    # Basic validation - model names are typically alphanumeric with dashes/dots
    if len(model_name) > 100:
        raise ValidationError("Model name must be 100 characters or less")
    
    return model_name.strip()


def validate_vendor(vendor: str) -> str:
    """Validate vendor name."""
    vendor = vendor.strip().lower()
    
    known_vendors = {
        'openai': 'openai',
        'anthropic': 'anthropic',
        'google': 'google',
        'mistral': 'mistral',
        'cohere': 'cohere',
        'meta': 'meta',
        'huggingface': 'huggingface',
        'local': 'local',
        'ollama': 'ollama',
    }
    
    if vendor in known_vendors:
        return known_vendors[vendor]
    
    # Allow unknown vendors with warning
    print(f"   ⚠️ Unknown vendor: {vendor}. Proceeding anyway...")
    return vendor


def validate_api_key(api_key: str) -> str:
    """Validate API key format."""
    if not api_key:
        raise ValidationError("API key cannot be empty")
    
    # Basic format checks
    if len(api_key) < 10:
        raise ValidationError("API key seems too short")
    
    if ' ' in api_key:
        raise ValidationError("API key should not contain spaces")
    
    return api_key.strip()


def validate_file_path(path_str: str, must_exist: bool = True, extensions: List[str] = None) -> Path:
    """
    Validate file path.
    
    Args:
        path_str: Path string to validate
        must_exist: Whether the file must exist
        extensions: Allowed file extensions (e.g., ['.csv', '.xlsx'])
    
    Returns:
        Validated Path object
    """
    path = Path(path_str.strip().strip('"').strip("'"))
    
    if must_exist and not path.exists():
        raise ValidationError(f"File not found: {path}")
    
    if extensions and path.suffix.lower() not in [e.lower() for e in extensions]:
        raise ValidationError(f"File must have extension: {', '.join(extensions)}")
    
    return path


def validate_integer(value_str: str, min_val: int = None, max_val: int = None) -> int:
    """Validate integer value with optional bounds."""
    try:
        value = int(value_str)
    except ValueError:
        raise ValidationError(f"Not a valid integer: {value_str}")
    
    if min_val is not None and value < min_val:
        raise ValidationError(f"Value must be >= {min_val}")
    
    if max_val is not None and value > max_val:
        raise ValidationError(f"Value must be <= {max_val}")
    
    return value


def validate_float(value_str: str, min_val: float = None, max_val: float = None) -> float:
    """Validate float value with optional bounds."""
    try:
        value = float(value_str)
    except ValueError:
        raise ValidationError(f"Not a valid number: {value_str}")
    
    if min_val is not None and value < min_val:
        raise ValidationError(f"Value must be >= {min_val}")
    
    if max_val is not None and value > max_val:
        raise ValidationError(f"Value must be <= {max_val}")
    
    return value


def validate_choice(value: str, choices: List[str], case_sensitive: bool = False) -> str:
    """Validate value is one of allowed choices."""
    if not case_sensitive:
        value = value.lower()
        choices = [c.lower() for c in choices]
    
    if value not in choices:
        raise ValidationError(f"Must be one of: {', '.join(choices)}")
    
    return value


def get_yes_no(prompt: str, default: bool = None) -> bool:
    """Get yes/no response from user."""
    suffix = " (y/n): " if default is None else f" ({'Y/n' if default else 'y/N'}): "
    
    while True:
        response = input(prompt + suffix).strip().lower()
        
        if not response and default is not None:
            return default
        
        if response in ['y', 'yes', 'true', '1']:
            return True
        elif response in ['n', 'no', 'false', '0']:
            return False
        else:
            print("   Please enter 'y' or 'n'")


def get_choice(prompt: str, choices: List[str], default: str = None) -> str:
    """Get choice from a list of options."""
    print(prompt)
    for i, choice in enumerate(choices, 1):
        default_marker = " (default)" if choice == default else ""
        print(f"   {i}. {choice}{default_marker}")
    
    while True:
        suffix = " (Enter for default): " if default else " (number or name): "
        response = input(f"\n   Enter choice{suffix}").strip()
        
        if not response and default:
            print(f"   → Using default: {default}")
            return default
        
        # Try as number
        try:
            idx = int(response) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
        except ValueError:
            pass
        
        # Try as name
        if response.lower() in [c.lower() for c in choices]:
            for c in choices:
                if c.lower() == response.lower():
                    return c
        
        print(f"   ❌ Invalid choice. Enter 1-{len(choices)} or the option name.")


def get_choice_with_skip(
    prompt: str, 
    choices: List[str], 
    default: str = None,
    allow_skip: bool = True,
    confirm: bool = True
) -> str:
    """
    Get choice from list with skip option and confirmation.
    
    Args:
        prompt: Prompt to display
        choices: List of choices
        default: Default choice if Enter pressed
        allow_skip: Whether to allow skipping (uses default)
        confirm: Whether to confirm the choice
    
    Returns:
        Selected choice
    """
    print(prompt)
    for i, choice in enumerate(choices, 1):
        default_marker = " ← default" if choice == default else ""
        print(f"   {i}. {choice}{default_marker}")
    
    if allow_skip and default:
        print(f"\n   Press Enter to use default ({default})")
    
    while True:
        response = input("\n   Your choice: ").strip()
        
        # Handle Enter for default
        if not response:
            if default:
                selected = default
                print(f"   → Using default: {selected}")
            else:
                print("   ❌ Please make a selection")
                continue
        else:
            # Try as number
            try:
                idx = int(response) - 1
                if 0 <= idx < len(choices):
                    selected = choices[idx]
                else:
                    print(f"   ❌ Enter 1-{len(choices)}")
                    continue
            except ValueError:
                # Try as name
                matched = None
                for c in choices:
                    if c.lower() == response.lower():
                        matched = c
                        break
                
                if matched:
                    selected = matched
                else:
                    print(f"   ❌ Invalid choice: {response}")
                    continue
        
        # Confirm if requested
        if confirm:
            confirmed, _ = confirm_input("Selection", selected)
            if confirmed:
                return selected
            else:
                print("   Let's try again...")
                continue
        else:
            return selected

