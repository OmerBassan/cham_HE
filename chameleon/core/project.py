"""
Project management for Chameleon.

Handles project creation, loading, and file management.
"""

import yaml
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd

from chameleon.core.schemas import (
    ProjectConfig, 
    Modality, 
    BackendType, 
    DistortionConfig,
    DataSchema,
    ProjectMetadata,
)


class Project:
    """Represents a Chameleon evaluation project."""
    
    CONFIG_FILE = "project_config.yaml"
    USER_CONFIG_FILE = "config.yaml"
    ENV_FILE = ".env"
    README_FILE = "README.md"
    GITIGNORE_FILE = ".gitignore"
    
    SUBDIRS = ["original_data", "distorted_data", "results", "analysis"]
    
    def __init__(self, config: ProjectConfig):
        """Initialize project from config."""
        self.config = config
        self.project_dir = Path(config.project_dir)
        self.original_data_dir = Path(config.original_data_dir)
        self.distorted_data_dir = Path(config.distorted_data_dir)
        self.results_dir = Path(config.results_dir)
        self.analysis_dir = Path(config.analysis_dir)
    
    @classmethod
    def create(
        cls,
        name: str,
        modality: Modality,
        model_name: str,
        backend_type: BackendType,
        base_dir: Path = Path("Projects"),
        description: Optional[str] = None,
        distortion_levels: Optional[List[float]] = None,
        distortions_per_question: int = 10,
    ) -> "Project":
        """
        Create a new project with directory structure.
        
        Args:
            name: Project name
            modality: Input modality (text, image, etc.)
            model_name: Target model name
            backend_type: Backend type for model
            base_dir: Base directory for projects
            description: Optional project description
            distortion_levels: List of miu values
            distortions_per_question: Number of distortions per question per miu
        
        Returns:
            Created Project instance
        """
        base_dir = Path(base_dir)
        project_dir = base_dir / name
        
        if project_dir.exists():
            raise FileExistsError(f"Project already exists: {project_dir}")
        
        # Create distortion config
        distortion_config = DistortionConfig(
            miu_values=distortion_levels or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            distortions_per_question=distortions_per_question,
        )
        
        # Create project config
        config = ProjectConfig(
            project_name=name,
            modality=modality,
            model_name=model_name,
            backend_type=backend_type,
            description=description,
            project_dir=str(project_dir),
            original_data_dir=str(project_dir / "original_data"),
            distorted_data_dir=str(project_dir / "distorted_data"),
            results_dir=str(project_dir / "results"),
            analysis_dir=str(project_dir / "analysis"),
            distortion_config=distortion_config,
        )
        
        project = cls(config)
        project._create_structure()
        project._save_config()
        project._create_readme()
        project._create_gitignore()
        
        return project
    
    def _create_structure(self):
        """Create project directory structure."""
        self.project_dir.mkdir(parents=True, exist_ok=True)
        
        for subdir in self.SUBDIRS:
            (self.project_dir / subdir).mkdir(exist_ok=True)
            # Create .gitkeep to track empty directories
            gitkeep = self.project_dir / subdir / ".gitkeep"
            gitkeep.touch()
    
    def _save_config(self):
        """Save project configuration to YAML."""
        config_path = self.project_dir / self.CONFIG_FILE
        
        # Use model_dump with mode='json' to get plain dict without Python objects
        config_dict = self.config.model_dump(mode='json')
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    def _create_readme(self):
        """Create project README."""
        readme_path = self.project_dir / self.README_FILE
        
        content = f"""# {self.config.project_name}

{self.config.description or 'Chameleon evaluation project.'}

## Configuration

- **Modality:** {self.config.modality}
- **Model:** {self.config.model_name}
- **Backend:** {self.config.backend_type}
- **Created:** {self.config.metadata.created_at}

## Distortion Settings

- **Miu Values:** {self.config.distortion_config.miu_values}
- **Distortions per Question:** {self.config.distortion_config.distortions_per_question}
- **Engine:** {self.config.distortion_config.engine}

## Directory Structure

```
{self.config.project_name}/
├── original_data/      # Original input data (CSV, JSON)
├── distorted_data/     # Distorted versions for each miu level
├── results/            # Model evaluation outputs
├── analysis/           # Analysis reports and visualizations
├── config.yaml         # User configuration (miu, distortions, etc.)
├── project_config.yaml # Full project configuration
├── .env                # API keys (git-ignored)
└── README.md
```

## Usage

1. Add original data to `original_data/`
2. Generate distortions: `python cli.py distort --project {self.config.project_name}`
3. Run evaluation: `python cli.py evaluate --project {self.config.project_name}`
4. Analyze results: `python cli.py analyze --project {self.config.project_name}`

## Files

See `project_config.yaml` for full configuration details.
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _create_gitignore(self):
        """Create .gitignore for project."""
        gitignore_path = self.project_dir / self.GITIGNORE_FILE
        
        content = """# Sensitive files
.env
*.key
*_secret*

# Python
__pycache__/
*.py[cod]
*$py.class

# Logs
*.log

# Large data files (optional - uncomment if needed)
# *.csv
# *.jsonl

# IDE
.vscode/
.idea/
"""
        
        with open(gitignore_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def create_env_file(self, api_keys: Dict[str, str]):
        """Create .env file with API keys."""
        env_path = self.project_dir / self.ENV_FILE
        
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(f"# API Keys for {self.config.project_name}\n")
            f.write(f"# Created: {datetime.now().isoformat()}\n\n")
            
            for key, value in api_keys.items():
                f.write(f"{key}={value}\n")
    
    def create_user_config(self, config: Dict[str, Any]):
        """Create user config.yaml file."""
        config_path = self.project_dir / self.USER_CONFIG_FILE
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    @classmethod
    def load(cls, project_path: Path) -> "Project":
        """
        Load an existing project from disk.
        
        Args:
            project_path: Path to project directory
        
        Returns:
            Loaded Project instance
        """
        project_path = Path(project_path)
        config_path = project_path / cls.CONFIG_FILE
        
        if not config_path.exists():
            raise FileNotFoundError(f"Project config not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        config = ProjectConfig(**config_dict)
        return cls(config)
    
    def load_original_data(self, filename: Optional[str] = None) -> pd.DataFrame:
        """Load original data from project."""
        return self._load_data_from_dir(self.original_data_dir, filename)
    
    def load_distorted_data(self, filename: Optional[str] = None) -> pd.DataFrame:
        """Load distorted data from project."""
        return self._load_data_from_dir(self.distorted_data_dir, filename)
    
    def load_results(self, filename: Optional[str] = None) -> pd.DataFrame:
        """Load results data from project."""
        return self._load_data_from_dir(self.results_dir, filename)
    
    def _load_data_from_dir(self, dir_path: Path, filename: Optional[str] = None) -> pd.DataFrame:
        """Load data from a directory."""
        if filename:
            file_path = dir_path / filename
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            return self._load_file(file_path)
        
        # Find first CSV or JSON file
        for ext in ['*.csv', '*.jsonl', '*.json']:
            files = list(dir_path.glob(ext))
            if files:
                # Use most recently modified
                latest = max(files, key=lambda p: p.stat().st_mtime)
                return self._load_file(latest)
        
        raise FileNotFoundError(f"No data files found in {dir_path}")
    
    def _load_file(self, file_path: Path) -> pd.DataFrame:
        """Load a single data file."""
        ext = file_path.suffix.lower()
        
        if ext == '.csv':
            return pd.read_csv(file_path, encoding='utf-8')
        elif ext == '.jsonl':
            return pd.read_json(file_path, lines=True)
        elif ext == '.json':
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def save_results(self, df: pd.DataFrame, filename: str):
        """Save results to project."""
        output_path = self.results_dir / filename
        
        if filename.endswith('.csv'):
            df.to_csv(output_path, index=False, encoding='utf-8')
        elif filename.endswith('.jsonl'):
            df.to_json(output_path, orient='records', lines=True)
        elif filename.endswith('.json'):
            df.to_json(output_path, orient='records')
        else:
            # Default to CSV
            output_path = output_path.with_suffix('.csv')
            df.to_csv(output_path, index=False, encoding='utf-8')
        
        return output_path
    
    def get_status(self) -> Dict[str, Any]:
        """Get project status information."""
        status = {
            "name": self.config.project_name,
            "exists": self.project_dir.exists(),
            "config": self.config.model_dump(),
            "files": {},
        }
        
        for subdir in self.SUBDIRS:
            subdir_path = self.project_dir / subdir
            if subdir_path.exists():
                files_by_ext = {}
                for f in subdir_path.iterdir():
                    if f.is_file() and not f.name.startswith('.'):
                        ext = f.suffix.lower() or 'no_ext'
                        files_by_ext[ext] = files_by_ext.get(ext, 0) + 1
                status["files"][subdir] = files_by_ext
            else:
                status["files"][subdir] = {}
        
        return status
    
    def get_user_config(self) -> Dict[str, Any]:
        """Load user config.yaml if exists."""
        config_path = self.project_dir / self.USER_CONFIG_FILE
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def update_user_config(self, updates: Dict[str, Any]):
        """Update user config.yaml."""
        current = self.get_user_config()
        
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        updated = deep_update(current, updates)
        self.create_user_config(updated)
        return updated


def list_projects(base_dir: Path = Path("Projects")) -> List[Dict[str, Any]]:
    """
    List all projects in the base directory.
    
    Args:
        base_dir: Base projects directory
    
    Returns:
        List of project info dictionaries
    """
    base_dir = Path(base_dir)
    projects = []
    
    if not base_dir.exists():
        return projects
    
    for item in base_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            config_path = item / Project.CONFIG_FILE
            
            project_info = {
                "name": item.name,
                "path": str(item),
                "exists": True,
            }
            
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_dict = yaml.safe_load(f)
                    project_info["config"] = config_dict
                    
                    # Get file counts
                    files = {}
                    for subdir in Project.SUBDIRS:
                        subdir_path = item / subdir
                        if subdir_path.exists():
                            count = sum(1 for f in subdir_path.iterdir() 
                                       if f.is_file() and not f.name.startswith('.'))
                            files[subdir] = {"total": count}
                    project_info["files"] = files
                    
                except Exception as e:
                    project_info["error"] = str(e)
            else:
                project_info["error"] = "No project_config.yaml found"
            
            projects.append(project_info)
    
    return sorted(projects, key=lambda p: p["name"])
