"""
OpenAI API backend for model inference.

Supports:
- GPT-4o, GPT-4o-mini, GPT-4, GPT-3.5-turbo, etc.
- Batch API for async processing
- Streaming responses
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator
from datetime import datetime

from chameleon.models.base import (
    ModelBackend, 
    BatchRequest, 
    BatchResponse, 
    BatchStatus,
    RequestStatus
)


class OpenAIBackend(ModelBackend):
    """OpenAI API backend."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenAI backend.
        
        Args:
            model_name: OpenAI model identifier
            api_key: API key (defaults to OPENAI_API_KEY env var)
            organization: Organization ID (optional)
            **kwargs: Additional configuration
        """
        super().__init__(model_name, **kwargs)
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.organization = organization or os.getenv("OPENAI_ORG_ID")
        self._client = None
    
    @property
    def backend_name(self) -> str:
        return "openai"
    
    @property
    def client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(
                    api_key=self.api_key,
                    organization=self.organization
                )
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
        return self._client
    
    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        if not self.api_key:
            return False
        try:
            import openai
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
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        return response.choices[0].message.content.strip()
    
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
            delay: Delay between requests (to avoid rate limits)
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
        Submit a batch for async processing via OpenAI Batch API.
        
        Args:
            requests: List of batch requests
            description: Description of the batch
            **kwargs: Additional parameters
        
        Returns:
            Batch ID
        """
        import tempfile
        
        # Create JSONL file with requests
        batch_requests = []
        for request in requests:
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})
            
            batch_requests.append({
                "custom_id": request.custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                }
            })
        
        # Write to temp file and upload
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for req in batch_requests:
                f.write(json.dumps(req) + '\n')
            temp_path = f.name
        
        try:
            # Upload file
            with open(temp_path, 'rb') as f:
                batch_file = self.client.files.create(file=f, purpose="batch")
            
            # Create batch
            batch = self.client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": description}
            )
            
            return batch.id
            
        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)
    
    def get_batch_status(self, batch_id: str) -> BatchStatus:
        """Get status of an async batch."""
        batch = self.client.batches.retrieve(batch_id)
        
        status_map = {
            "validating": RequestStatus.PENDING,
            "in_progress": RequestStatus.IN_PROGRESS,
            "finalizing": RequestStatus.IN_PROGRESS,
            "completed": RequestStatus.COMPLETED,
            "failed": RequestStatus.FAILED,
            "cancelled": RequestStatus.CANCELLED,
            "expired": RequestStatus.FAILED,
        }
        
        counts = batch.request_counts if batch.request_counts else None
        
        return BatchStatus(
            batch_id=batch_id,
            status=status_map.get(batch.status, RequestStatus.PENDING),
            total_requests=counts.total if counts else 0,
            completed_requests=counts.completed if counts else 0,
            failed_requests=counts.failed if counts else 0,
            created_at=str(batch.created_at) if batch.created_at else None,
            completed_at=str(batch.completed_at) if hasattr(batch, 'completed_at') and batch.completed_at else None,
            output_file_id=getattr(batch, 'output_file_id', None),
            error=str(batch.errors) if batch.errors else None,
        )
    
    def get_batch_results(self, batch_id: str) -> List[BatchResponse]:
        """Get results from a completed batch."""
        status = self.get_batch_status(batch_id)
        
        if status.status != RequestStatus.COMPLETED:
            raise ValueError(f"Batch not completed. Status: {status.status}")
        
        if not status.output_file_id:
            raise ValueError("No output file available")
        
        # Download results
        content = self.client.files.content(status.output_file_id)
        
        responses = []
        for line in content.text.strip().split('\n'):
            if not line:
                continue
            
            result = json.loads(line)
            custom_id = result.get('custom_id', '')
            
            if result.get('error'):
                responses.append(BatchResponse(
                    custom_id=custom_id,
                    content="",
                    success=False,
                    error=str(result['error']),
                ))
            else:
                try:
                    response_content = result['response']['body']['choices'][0]['message']['content']
                    usage = result['response']['body'].get('usage', {})
                    
                    responses.append(BatchResponse(
                        custom_id=custom_id,
                        content=response_content.strip(),
                        success=True,
                        usage={
                            "input_tokens": usage.get('prompt_tokens', 0),
                            "output_tokens": usage.get('completion_tokens', 0),
                        },
                        raw_response=result,
                    ))
                except (KeyError, IndexError) as e:
                    responses.append(BatchResponse(
                        custom_id=custom_id,
                        content="",
                        success=False,
                        error=f"Failed to parse response: {e}",
                    ))
        
        return responses
    
    def cancel_batch(self, batch_id: str) -> bool:
        """Cancel an in-progress batch."""
        try:
            self.client.batches.cancel(batch_id)
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
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            **kwargs
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


