"""
Anthropic API backend for model inference.

Supports:
- Claude 3.5 Sonnet, Claude 3 Opus, etc.
- Batch API for async processing (Message Batches)
- Streaming responses
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Iterator

from chameleon.models.base import (
    ModelBackend,
    BatchRequest,
    BatchResponse,
    BatchStatus,
    RequestStatus
)


class AnthropicBackend(ModelBackend):
    """Anthropic API backend."""
    
    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Anthropic backend.
        
        Args:
            model_name: Anthropic model identifier
            api_key: API key (defaults to ANTHROPIC_API_KEY env var)
            **kwargs: Additional configuration
        """
        super().__init__(model_name, **kwargs)
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None
    
    @property
    def backend_name(self) -> str:
        return "anthropic"
    
    @property
    def client(self):
        """Lazy load Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        return self._client
    
    def is_available(self) -> bool:
        """Check if Anthropic API is available."""
        if not self.api_key:
            return False
        try:
            import anthropic
            return True
        except ImportError:
            return False
    
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """
        Generate a single completion.
        
        Args:
            prompt: User prompt
            system_prompt: System/instruction prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters
        
        Returns:
            Generated text
        """
        messages = [{"role": "user", "content": prompt}]
        
        create_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        if system_prompt:
            create_kwargs["system"] = system_prompt
        
        response = self.client.messages.create(**create_kwargs, **kwargs)
        
        return response.content[0].text.strip()
    
    def complete_batch(
        self,
        requests: List[BatchRequest],
        delay: float = 0.1,
        **kwargs
    ) -> List[BatchResponse]:
        """
        Process a batch of requests synchronously.
        
        Args:
            requests: List of batch requests
            delay: Delay between requests
            **kwargs: Additional parameters
        
        Returns:
            List of batch responses
        """
        responses = []
        
        for request in requests:
            try:
                content = self.complete(
                    prompt=request.prompt,
                    system_prompt=request.system_prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    **kwargs
                )
                
                responses.append(BatchResponse(
                    custom_id=request.custom_id,
                    content=content,
                    success=True,
                ))
                
            except Exception as e:
                responses.append(BatchResponse(
                    custom_id=request.custom_id,
                    content="",
                    success=False,
                    error=str(e),
                ))
            
            if delay > 0:
                time.sleep(delay)
        
        return responses
    
    def supports_async_batch(self) -> bool:
        return True
    
    def submit_batch(
        self,
        requests: List[BatchRequest],
        description: str = "",
        **kwargs
    ) -> str:
        """
        Submit a batch for async processing via Anthropic Message Batches API.
        
        Args:
            requests: List of batch requests
            description: Description of the batch
            **kwargs: Additional parameters
        
        Returns:
            Batch ID
        """
        # Build batch requests
        batch_requests = []
        
        for request in requests:
            params = {
                "model": self.model_name,
                "max_tokens": request.max_tokens,
                "messages": [{"role": "user", "content": request.prompt}],
            }
            
            if request.system_prompt:
                params["system"] = request.system_prompt
            
            batch_requests.append({
                "custom_id": request.custom_id,
                "params": params,
            })
        
        # Create message batch
        message_batch = self.client.messages.batches.create(requests=batch_requests)
        
        return message_batch.id
    
    def get_batch_status(self, batch_id: str) -> BatchStatus:
        """Get status of an async batch."""
        batch = self.client.messages.batches.retrieve(batch_id)
        
        status_map = {
            "in_progress": RequestStatus.IN_PROGRESS,
            "ended": RequestStatus.COMPLETED,
            "canceling": RequestStatus.CANCELLED,
        }
        
        counts = batch.request_counts if hasattr(batch, 'request_counts') else None
        
        # Determine actual status
        if batch.processing_status == "ended":
            if counts and counts.errored > 0:
                status = RequestStatus.COMPLETED  # Completed with some errors
            else:
                status = RequestStatus.COMPLETED
        else:
            status = status_map.get(batch.processing_status, RequestStatus.PENDING)
        
        return BatchStatus(
            batch_id=batch_id,
            status=status,
            total_requests=counts.processing + counts.succeeded + counts.errored if counts else 0,
            completed_requests=counts.succeeded if counts else 0,
            failed_requests=counts.errored if counts else 0,
            created_at=str(batch.created_at) if hasattr(batch, 'created_at') else None,
            completed_at=str(batch.ended_at) if hasattr(batch, 'ended_at') and batch.ended_at else None,
        )
    
    def get_batch_results(self, batch_id: str) -> List[BatchResponse]:
        """Get results from a completed batch."""
        responses = []
        
        # Iterate through results
        for result in self.client.messages.batches.results(batch_id):
            custom_id = result.custom_id
            
            if result.result.type == "succeeded":
                message = result.result.message
                content = message.content[0].text if message.content else ""
                usage = {
                    "input_tokens": message.usage.input_tokens if message.usage else 0,
                    "output_tokens": message.usage.output_tokens if message.usage else 0,
                }
                
                responses.append(BatchResponse(
                    custom_id=custom_id,
                    content=content.strip(),
                    success=True,
                    usage=usage,
                ))
            else:
                error_msg = str(result.result.error) if hasattr(result.result, 'error') else "Unknown error"
                responses.append(BatchResponse(
                    custom_id=custom_id,
                    content="",
                    success=False,
                    error=error_msg,
                ))
        
        return responses
    
    def cancel_batch(self, batch_id: str) -> bool:
        """Cancel an in-progress batch."""
        try:
            self.client.messages.batches.cancel(batch_id)
            return True
        except Exception:
            return False
    
    def supports_streaming(self) -> bool:
        return True
    
    def complete_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.0,
        **kwargs
    ) -> Iterator[str]:
        """Generate a streaming completion."""
        messages = [{"role": "user", "content": prompt}]
        
        create_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        if system_prompt:
            create_kwargs["system"] = system_prompt
        
        with self.client.messages.stream(**create_kwargs, **kwargs) as stream:
            for text in stream.text_stream:
                yield text


