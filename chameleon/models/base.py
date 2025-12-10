"""
Base model backend interface.

Defines the abstract interface that all model backends must implement,
enabling consistent usage across OpenAI, Anthropic, MLX, CUDA, etc.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Iterator
from enum import Enum


class RequestStatus(str, Enum):
    """Status of a batch request."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchRequest:
    """A single request in a batch."""
    custom_id: str
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: int = 100
    temperature: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResponse:
    """Response to a single batch request."""
    custom_id: str
    content: str
    success: bool = True
    error: Optional[str] = None
    usage: Optional[Dict[str, int]] = None  # {"input_tokens": X, "output_tokens": Y}
    raw_response: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchStatus:
    """Status of a batch job."""
    batch_id: str
    status: RequestStatus
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    output_file_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def progress(self) -> float:
        """Progress as percentage (0-100)."""
        if self.total_requests == 0:
            return 0.0
        return (self.completed_requests / self.total_requests) * 100


class ModelBackend(ABC):
    """
    Abstract base class for model backends.
    
    All backends (OpenAI, Anthropic, MLX, CUDA, etc.) must implement this interface.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the backend.
        
        Args:
            model_name: Name/identifier of the model to use
            **kwargs: Backend-specific configuration
        """
        self.model_name = model_name
        self.config = kwargs
    
    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the name of this backend (e.g., 'openai', 'anthropic')."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available (API keys set, hardware present, etc.)."""
        pass
    
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """
        Generate a single completion.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def complete_batch(
        self,
        requests: List[BatchRequest],
        **kwargs
    ) -> List[BatchResponse]:
        """
        Process a batch of requests synchronously.
        
        Args:
            requests: List of batch requests
            **kwargs: Additional parameters
        
        Returns:
            List of batch responses
        """
        pass
    
    # Optional async batch API (for providers that support it)
    def supports_async_batch(self) -> bool:
        """Check if this backend supports async batch processing."""
        return False
    
    def submit_batch(
        self,
        requests: List[BatchRequest],
        description: str = "",
        **kwargs
    ) -> str:
        """
        Submit a batch for async processing.
        
        Args:
            requests: List of batch requests
            description: Description of the batch
            **kwargs: Additional parameters
        
        Returns:
            Batch ID for tracking
        """
        raise NotImplementedError("This backend doesn't support async batch processing")
    
    def get_batch_status(self, batch_id: str) -> BatchStatus:
        """
        Get the status of an async batch.
        
        Args:
            batch_id: ID of the batch to check
        
        Returns:
            BatchStatus with current progress
        """
        raise NotImplementedError("This backend doesn't support async batch processing")
    
    def get_batch_results(self, batch_id: str) -> List[BatchResponse]:
        """
        Get results from a completed async batch.
        
        Args:
            batch_id: ID of the completed batch
        
        Returns:
            List of batch responses
        """
        raise NotImplementedError("This backend doesn't support async batch processing")
    
    def cancel_batch(self, batch_id: str) -> bool:
        """
        Cancel an in-progress batch.
        
        Args:
            batch_id: ID of the batch to cancel
        
        Returns:
            True if cancellation was successful
        """
        raise NotImplementedError("This backend doesn't support async batch processing")
    
    # Streaming support (optional)
    def supports_streaming(self) -> bool:
        """Check if this backend supports streaming responses."""
        return False
    
    def complete_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """
        Generate a streaming completion.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters
        
        Yields:
            Text chunks as they're generated
        """
        raise NotImplementedError("This backend doesn't support streaming")
    
    def format_mcq_prompt(
        self,
        question: str,
        choices: Dict[str, str],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Format a multiple-choice question for this backend.
        
        Args:
            question: The question text
            choices: Dict of choice labels to choice text
            system_prompt: Optional system/instruction prompt
        
        Returns:
            Formatted prompt string
        """
        choices_text = "\n".join([f"{k}: {v}" for k, v in sorted(choices.items())])
        
        default_system = (
            "Answer the following multiple choice question. "
            "Respond with ONLY the letter of the correct answer (A, B, C, or D). "
            "No explanations."
        )
        
        if system_prompt:
            prompt = f"{system_prompt}\n\nQuestion: {question}\n\nOptions:\n{choices_text}\n\nAnswer:"
        else:
            prompt = f"{default_system}\n\nQuestion: {question}\n\nOptions:\n{choices_text}\n\nAnswer:"
        
        return prompt
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"


