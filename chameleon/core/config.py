"""
Global configuration management for the Chameleon framework.

Handles:
- Loading/saving global configuration
- Environment variable management
- API key handling
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from dataclasses import dataclass, field


@dataclass
class ChameleonConfig:
    """Global configuration for the Chameleon framework."""
    
    # Base paths
    projects_dir: Path = field(default_factory=lambda: Path("projects"))
    config_dir: Path = field(default_factory=lambda: Path("config"))
    
    # API Keys (loaded from environment)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Default batch processing settings
    default_batch_size: int = 50
    default_max_retries: int = 3
    default_retry_delay: float = 1.0
    default_request_timeout: int = 600
    
    # Local model settings
    mlx_model_path: Optional[str] = None
    cuda_device: str = "cuda:0"
    
    # Distortion server settings
    distortion_server_url: str = "http://localhost:8000"
    distortion_server_timeout: int = 3600
    
    def __post_init__(self):
        """Load API keys from environment after initialization."""
        self.openai_api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        
        # Ensure paths are Path objects
        if isinstance(self.projects_dir, str):
            self.projects_dir = Path(self.projects_dir)
        if isinstance(self.config_dir, str):
            self.config_dir = Path(self.config_dir)
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "ChameleonConfig":
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path("config/chameleon_config.yaml")
        
        if not config_path.exists():
            # Return default configuration
            return cls()
        
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f) or {}
        
        return cls(
            projects_dir=Path(data.get("projects_dir", "projects")),
            config_dir=Path(data.get("config_dir", "config")),
            openai_api_key=data.get("openai_api_key") or os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=data.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY"),
            default_batch_size=data.get("default_batch_size", 50),
            default_max_retries=data.get("default_max_retries", 3),
            default_retry_delay=data.get("default_retry_delay", 1.0),
            default_request_timeout=data.get("default_request_timeout", 600),
            mlx_model_path=data.get("mlx_model_path"),
            cuda_device=data.get("cuda_device", "cuda:0"),
            distortion_server_url=data.get("distortion_server_url", "http://localhost:8000"),
            distortion_server_timeout=data.get("distortion_server_timeout", 3600),
        )
    
    def save(self, config_path: Optional[Path] = None) -> None:
        """Save configuration to YAML file."""
        if config_path is None:
            config_path = self.config_dir / "chameleon_config.yaml"
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "projects_dir": str(self.projects_dir),
            "config_dir": str(self.config_dir),
            "default_batch_size": self.default_batch_size,
            "default_max_retries": self.default_max_retries,
            "default_retry_delay": self.default_retry_delay,
            "default_request_timeout": self.default_request_timeout,
            "cuda_device": self.cuda_device,
            "distortion_server_url": self.distortion_server_url,
            "distortion_server_timeout": self.distortion_server_timeout,
        }
        
        # Don't save API keys to file - use environment variables
        if self.mlx_model_path:
            data["mlx_model_path"] = self.mlx_model_path
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "projects_dir": str(self.projects_dir),
            "config_dir": str(self.config_dir),
            "has_openai_key": bool(self.openai_api_key),
            "has_anthropic_key": bool(self.anthropic_api_key),
            "default_batch_size": self.default_batch_size,
            "default_max_retries": self.default_max_retries,
            "default_retry_delay": self.default_retry_delay,
            "default_request_timeout": self.default_request_timeout,
            "mlx_model_path": self.mlx_model_path,
            "cuda_device": self.cuda_device,
            "distortion_server_url": self.distortion_server_url,
        }
    
    def validate(self) -> Dict[str, bool]:
        """Validate configuration and return status of each component."""
        status = {
            "projects_dir_exists": self.projects_dir.exists(),
            "config_dir_exists": self.config_dir.exists(),
            "openai_api_key_set": bool(self.openai_api_key),
            "anthropic_api_key_set": bool(self.anthropic_api_key),
        }
        
        # Check MLX availability (Apple Silicon)
        try:
            import mlx
            status["mlx_available"] = True
        except ImportError:
            status["mlx_available"] = False
        
        # Check CUDA availability
        try:
            import torch
            status["cuda_available"] = torch.cuda.is_available()
        except ImportError:
            status["cuda_available"] = False
        
        return status

