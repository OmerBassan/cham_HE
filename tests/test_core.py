"""Tests for core module functionality."""

import pytest
import tempfile
from pathlib import Path
import shutil

from chameleon.core.schemas import (
    Modality,
    BackendType,
    ProjectConfig,
    QuestionItem,
    ResultItem,
)
from chameleon.core.project import Project, list_projects
from chameleon.core.config import ChameleonConfig


class TestSchemas:
    """Tests for data schemas."""
    
    def test_modality_enum(self):
        """Test Modality enum values."""
        assert Modality.TEXT.value == "text"
        assert Modality.IMAGE.value == "image"
        assert Modality.AUDIO.value == "audio"
        assert Modality.VIDEO.value == "video"
    
    def test_backend_type_enum(self):
        """Test BackendType enum values."""
        assert BackendType.OPENAI.value == "openai"
        assert BackendType.ANTHROPIC.value == "anthropic"
        assert BackendType.MLX.value == "mlx"
        assert BackendType.CUDA_LOCAL.value == "cuda_local"
    
    def test_project_config_creation(self):
        """Test ProjectConfig creation."""
        config = ProjectConfig(
            project_name="test_project",
            modality=Modality.TEXT,
            model_name="gpt-4o",
            backend_type=BackendType.OPENAI,
        )
        
        assert config.project_name == "test_project"
        assert config.modality == Modality.TEXT
        assert config.model_name == "gpt-4o"
        assert config.backend_type == BackendType.OPENAI
    
    def test_project_config_to_dict(self):
        """Test ProjectConfig serialization."""
        config = ProjectConfig(
            project_name="test_project",
            modality=Modality.TEXT,
        )
        
        data = config.to_dict()
        
        assert data["project_name"] == "test_project"
        assert data["modality"] == "text"
        assert "paths" in data
        assert "metadata" in data
    
    def test_project_config_from_dict(self):
        """Test ProjectConfig deserialization."""
        data = {
            "project_name": "loaded_project",
            "modality": "image",
            "model_name": "claude-3",
            "backend_type": "anthropic",
        }
        
        config = ProjectConfig.from_dict(data)
        
        assert config.project_name == "loaded_project"
        assert config.modality == Modality.IMAGE
        assert config.model_name == "claude-3"
        assert config.backend_type == BackendType.ANTHROPIC
    
    def test_question_item(self):
        """Test QuestionItem creation."""
        item = QuestionItem(
            question_id="q_001",
            original_text="What is 2+2?",
            answer_options={"A": "3", "B": "4", "C": "5", "D": "6"},
            correct_answer="B",
        )
        
        assert item.question_id == "q_001"
        assert item.correct_answer == "B"
        assert item.distortion_level == 0.0
    
    def test_result_item(self):
        """Test ResultItem creation."""
        item = ResultItem(
            question_id="q_001",
            model_answer="B",
            is_correct=True,
        )
        
        assert item.question_id == "q_001"
        assert item.model_answer == "B"
        assert item.is_correct is True


class TestProject:
    """Tests for Project management."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_create_project(self, temp_dir):
        """Test project creation."""
        project = Project.create(
            name="test_project",
            modality=Modality.TEXT,
            model_name="gpt-4o",
            backend_type=BackendType.OPENAI,
            base_dir=temp_dir,
            description="Test project for unit tests",
        )
        
        # Check project was created
        assert project.exists()
        assert project.project_dir.exists()
        assert project.original_data_dir.exists()
        assert project.distorted_data_dir.exists()
        assert project.results_dir.exists()
        assert project.analysis_dir.exists()
        assert project.config_path.exists()
    
    def test_load_project(self, temp_dir):
        """Test loading an existing project."""
        # Create project first
        Project.create(
            name="loadable_project",
            modality=Modality.IMAGE,
            model_name="claude-3",
            backend_type=BackendType.ANTHROPIC,
            base_dir=temp_dir,
        )
        
        # Load it back
        project = Project.load(temp_dir / "loadable_project")
        
        assert project.name == "loadable_project"
        assert project.config.modality == Modality.IMAGE
        assert project.config.model_name == "claude-3"
        assert project.config.backend_type == BackendType.ANTHROPIC
    
    def test_project_status(self, temp_dir):
        """Test getting project status."""
        project = Project.create(
            name="status_test",
            modality=Modality.TEXT,
            base_dir=temp_dir,
        )
        
        status = project.get_status()
        
        assert status["name"] == "status_test"
        assert status["exists"] is True
        assert "config" in status
        assert "files" in status
    
    def test_list_projects(self, temp_dir):
        """Test listing multiple projects."""
        # Create a few projects
        Project.create(name="project_a", modality=Modality.TEXT, base_dir=temp_dir)
        Project.create(name="project_b", modality=Modality.IMAGE, base_dir=temp_dir)
        
        projects = list_projects(temp_dir)
        
        assert len(projects) == 2
        names = {p["name"] for p in projects}
        assert "project_a" in names
        assert "project_b" in names
    
    def test_project_delete(self, temp_dir):
        """Test project deletion."""
        project = Project.create(
            name="deletable_project",
            modality=Modality.TEXT,
            base_dir=temp_dir,
        )
        
        assert project.exists()
        
        # Delete without confirm should raise
        with pytest.raises(ValueError):
            project.delete(confirm=False)
        
        # Delete with confirm should work
        result = project.delete(confirm=True)
        
        assert result is True
        assert not project.project_dir.exists()


class TestConfig:
    """Tests for configuration management."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ChameleonConfig()
        
        assert config.projects_dir == Path("projects")
        assert config.default_batch_size == 50
        assert config.default_max_retries == 3
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = ChameleonConfig()
        
        validation = config.validate()
        
        assert "openai_api_key_set" in validation
        assert "anthropic_api_key_set" in validation
        assert isinstance(validation["openai_api_key_set"], bool)
    
    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = ChameleonConfig()
        
        data = config.to_dict()
        
        assert "projects_dir" in data
        assert "has_openai_key" in data
        assert "default_batch_size" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


