"""
Dummy backend for testing.

Returns predictable responses without making any API calls.
Useful for testing the framework without incurring costs.
"""

import random
import time
from typing import Dict, List, Optional, Any, Iterator

from chameleon.models.base import (
    ModelBackend,
    BatchRequest,
    BatchResponse,
    BatchStatus,
    RequestStatus
)


class DummyBackend(ModelBackend):
    """Dummy backend for testing."""
    
    def __init__(
        self,
        model_name: str = "dummy-model",
        response_delay: float = 0.01,
        random_answers: bool = False,
        default_answer: str = "A",
        error_rate: float = 0.0,
        **kwargs
    ):
        """
        Initialize dummy backend.
        
        Args:
            model_name: Model identifier (ignored)
            response_delay: Delay in seconds per response
            random_answers: Return random A/B/C/D answers
            default_answer: Default answer to return
            error_rate: Fraction of requests to fail (0.0 to 1.0)
            **kwargs: Additional configuration
        """
        super().__init__(model_name, **kwargs)
        
        self.response_delay = response_delay
        self.random_answers = random_answers
        self.default_answer = default_answer
        self.error_rate = error_rate
        
        self._batch_counter = 0
        self._batches: Dict[str, dict] = {}
    
    @property
    def backend_name(self) -> str:
        return "dummy"
    
    def is_available(self) -> bool:
        return True  # Always available
    
    def _generate_answer(self) -> str:
        """Generate a response."""
        if self.random_answers:
            return random.choice(["A", "B", "C", "D"])
        return self.default_answer
    
    def _should_fail(self) -> bool:
        """Determine if this request should fail."""
        return random.random() < self.error_rate
    
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """Generate a dummy completion."""
        if self.response_delay > 0:
            time.sleep(self.response_delay)
        
        if self._should_fail():
            raise RuntimeError("Simulated error")
        
        return self._generate_answer()
    
    def complete_batch(
        self,
        requests: List[BatchRequest],
        **kwargs
    ) -> List[BatchResponse]:
        """Process a batch of dummy requests."""
        responses = []
        
        for request in requests:
            if self.response_delay > 0:
                time.sleep(self.response_delay)
            
            if self._should_fail():
                responses.append(BatchResponse(
                    custom_id=request.custom_id,
                    content="",
                    success=False,
                    error="Simulated error",
                ))
            else:
                responses.append(BatchResponse(
                    custom_id=request.custom_id,
                    content=self._generate_answer(),
                    success=True,
                    usage={"input_tokens": 10, "output_tokens": 1},
                ))
        
        return responses
    
    def supports_async_batch(self) -> bool:
        return True
    
    def submit_batch(
        self,
        requests: List[BatchRequest],
        description: str = "",
        **kwargs
    ) -> str:
        """Submit a dummy batch."""
        self._batch_counter += 1
        batch_id = f"dummy_batch_{self._batch_counter}"
        
        # Store batch info
        self._batches[batch_id] = {
            "requests": requests,
            "description": description,
            "status": RequestStatus.IN_PROGRESS,
            "created_at": time.time(),
        }
        
        return batch_id
    
    def get_batch_status(self, batch_id: str) -> BatchStatus:
        """Get status of a dummy batch."""
        if batch_id not in self._batches:
            raise ValueError(f"Unknown batch: {batch_id}")
        
        batch = self._batches[batch_id]
        
        # Simulate completion after a short time
        elapsed = time.time() - batch["created_at"]
        if elapsed > 1.0:  # Complete after 1 second
            batch["status"] = RequestStatus.COMPLETED
        
        return BatchStatus(
            batch_id=batch_id,
            status=batch["status"],
            total_requests=len(batch["requests"]),
            completed_requests=len(batch["requests"]) if batch["status"] == RequestStatus.COMPLETED else 0,
            failed_requests=0,
        )
    
    def get_batch_results(self, batch_id: str) -> List[BatchResponse]:
        """Get results from a dummy batch."""
        if batch_id not in self._batches:
            raise ValueError(f"Unknown batch: {batch_id}")
        
        batch = self._batches[batch_id]
        
        if batch["status"] != RequestStatus.COMPLETED:
            raise ValueError("Batch not completed")
        
        return self.complete_batch(batch["requests"])
    
    def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a dummy batch."""
        if batch_id in self._batches:
            self._batches[batch_id]["status"] = RequestStatus.CANCELLED
            return True
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
        """Generate a dummy streaming completion."""
        answer = self._generate_answer()
        
        for char in answer:
            if self.response_delay > 0:
                time.sleep(self.response_delay)
            yield char


