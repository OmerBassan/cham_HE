"""
MLX backend for Apple Silicon inference.

Supports local model inference on Apple Silicon Macs using the MLX framework.
Models must be in MLX format or converted from HuggingFace.

Note: This is a stub implementation. Full implementation requires:
- MLX installed: pip install mlx mlx-lm
- Model downloaded in MLX format
"""

import os
from typing import Dict, List, Optional, Any, Iterator

from chameleon.models.base import (
    ModelBackend,
    BatchRequest,
    BatchResponse,
)


class MLXBackend(ModelBackend):
    """MLX backend for Apple Silicon."""
    
    def __init__(
        self,
        model_name: str = "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        model_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize MLX backend.
        
        Args:
            model_name: Model identifier (HuggingFace style or local path)
            model_path: Local path to model files (optional)
            **kwargs: Additional configuration
        """
        super().__init__(model_name, **kwargs)
        
        self.model_path = model_path or model_name
        self._model = None
        self._tokenizer = None
    
    @property
    def backend_name(self) -> str:
        return "mlx"
    
    def is_available(self) -> bool:
        """Check if MLX is available."""
        try:
            import mlx
            import mlx.core as mx
            # Check if running on Apple Silicon
            return mx.metal.is_available()
        except ImportError:
            return False
    
    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return
        
        try:
            from mlx_lm import load, generate
            
            self._model, self._tokenizer = load(self.model_path)
            self._generate_fn = generate
            
        except ImportError:
            raise ImportError(
                "MLX LM not installed. Run: pip install mlx mlx-lm\n"
                "Note: MLX only works on Apple Silicon Macs."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """
        Generate a completion using MLX.
        
        Args:
            prompt: User prompt
            system_prompt: System/instruction prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters
        
        Returns:
            Generated text
        """
        self._load_model()
        
        from mlx_lm import generate
        
        # Format prompt
        if system_prompt:
            full_prompt = f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]"
        else:
            full_prompt = f"<s>[INST] {prompt} [/INST]"
        
        # Generate
        response = generate(
            self._model,
            self._tokenizer,
            prompt=full_prompt,
            max_tokens=max_tokens,
            temp=temperature if temperature > 0 else 0.0,
            **kwargs
        )
        
        return response.strip()
    
    def complete_batch(
        self,
        requests: List[BatchRequest],
        **kwargs
    ) -> List[BatchResponse]:
        """
        Process a batch of requests.
        
        Note: MLX processes requests sequentially. For true batching,
        consider using the async API.
        
        Args:
            requests: List of batch requests
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
        
        return responses
    
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
        """Generate a streaming completion using MLX."""
        self._load_model()
        
        from mlx_lm import stream_generate
        
        # Format prompt
        if system_prompt:
            full_prompt = f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]"
        else:
            full_prompt = f"<s>[INST] {prompt} [/INST]"
        
        # Stream generate
        for token in stream_generate(
            self._model,
            self._tokenizer,
            prompt=full_prompt,
            max_tokens=max_tokens,
            temp=temperature if temperature > 0 else 0.0,
        ):
            yield token


