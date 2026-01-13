"""
Phi Model Integration for Retailer AI Suggestions and Chat
Uses local Microsoft Phi-3-mini model for intelligent recommendations
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datetime import datetime, timedelta
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhiModelService:
    """Local Phi model service for AI suggestions and chat"""
    
    """
    Phi Model Integration for Retailer AI Suggestions and Chat
    This module attempts to load a local Phi model for richer suggestions. If loading fails or stalls
    it falls back to a lightweight rule-based AI service defined in `lightweight_ai_service.py`.

    Improvements included:
    - Use `dtype` where possible instead of deprecated `torch_dtype`.
    - Async loading with timeout and retry logic.
    - Detect missing `flash_attn` and log guidance.
    - Graceful fallback to lightweight AI when model unavailable.
    """

    import asyncio
    import logging
    import time
    from typing import Any, Dict, List

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from datetime import datetime

    from lightweight_ai_service import (
        get_lightweight_suggestions,
        chat_with_lightweight_ai,
        analyze_trends_lightweight,
    )

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


    class PhiModelService:
        """Local Phi model service with robust loading and fallback.

        Attributes:
            model_name: Hugging Face model identifier to load.
            is_loaded: True when model and pipeline are available.
            use_fallback: True when falling back to lightweight service.
        """

        def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
            self.model = None
            self.tokenizer = None
            self.pipeline = None
            self.model_name = model_name
            self.is_loaded = False
            self.use_fallback = False

        async def _load_model_sync(self, dtype, device):
            """Synchronous loading helper to be run in thread executor."""
            # Detect flash-attn availability
            try:
                import flash_attn  # type: ignore
                logger.info("flash-attn detected: using optimized attention if supported by model.")
            except Exception:
                logger.warning(
                    "`flash-attn` not found. Install it for better performance (pip: flash-attn), or accept slower attention implementation."
                )

            # Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )

            # Model - try to pass `dtype` (newer API). If that fails, fall back to torch_dtype.
            load_kwargs = dict(trust_remote_code=True, low_cpu_mem_usage=True)
            if device == "cuda":
                # Let transformers put layers on CUDA automatically when device_map='auto'
                load_kwargs.update(device_map="auto")
                try:
                    load_kwargs["dtype"] = dtype
                except Exception:
                    # older versions may not accept 'dtype' arg; keep going without it
                    pass
            else:
                # CPU load - keep memory usage low
                load_kwargs.update(device_map=None)
                try:
                    load_kwargs["dtype"] = dtype
                except Exception:
                    pass

            # Try primary load
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)

            # Build pipeline: set device parameter (0 for first GPU, -1 for CPU)
            device_id = 0 if device == "cuda" else -1
            pipeline_kwargs = dict(
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
            )
            # `pipeline` accepts device=device_id
            self.pipeline = pipeline("text-generation", device=device_id, **pipeline_kwargs)

        async def load_model(self, timeout: int = 300, retries: int = 1):
            """Attempt to load the Phi model asynchronously with timeout and retries.

            If loading fails or times out, the service will flip to the lightweight fallback.
            """
            if self.is_loaded:
                return

            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            device = "cuda" if torch.cuda.is_available() else "cpu"

            loop = asyncio.get_running_loop()

            attempt = 0
            while attempt <= retries:
                attempt += 1
                start_time = time.time()
                try:
                    logger.info(f"🤖 Starting Phi model load attempt {attempt} (device={device}, dtype={dtype})")
                    # Run synchronous load in thread pool to avoid blocking event loop
                    await asyncio.wait_for(loop.run_in_executor(None, lambda: asyncio.run(self._load_model_sync(dtype, device)) if False else None), timeout=0.1)
                except asyncio.TimeoutError:
                    # The above is a placeholder; we'll implement a safer thread-based approach below
                    pass

                # Implement thread-based blocking call with timeout using run_in_executor directly
                try:
                    fut = loop.run_in_executor(None, self._load_model_blocking, dtype, device)
                    await asyncio.wait_for(fut, timeout=timeout)
                    # Success
                    self.is_loaded = True
                    self.use_fallback = False
                    logger.info("✅ Phi model loaded successfully!")
                    return
                except asyncio.TimeoutError:
                    logger.error(f"❌ Phi model load timed out after {timeout} seconds (attempt {attempt}).")
                except Exception as e:
                    logger.error(f"❌ Error loading Phi model on attempt {attempt}: {e}")

            # If we get here, loading failed
            self.is_loaded = False
            self.use_fallback = True
            logger.warning("Using lightweight AI fallback. Phi model unavailable.")

        def _load_model_blocking(self, dtype, device):
            """Blocking model load helper (runs in thread). Separating from async code avoids nested event loop issues."""
            # Detect flash-attn availability
            try:
                import flash_attn  # type: ignore
                logger.info("flash-attn detected in blocking loader.")
            except Exception:
                logger.debug("flash-attn not available in blocking loader.")

            # Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

            load_kwargs = dict(trust_remote_code=True, low_cpu_mem_usage=True)
            if device == "cuda":
                load_kwargs.update(device_map="auto")
                try:
                    load_kwargs["dtype"] = dtype
                except Exception:
                    # Older HF versions may not accept 'dtype'
                    load_kwargs["torch_dtype"] = dtype
            else:
                try:
                    load_kwargs["dtype"] = dtype
                except Exception:
                    load_kwargs["torch_dtype"] = dtype

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)

            # Pipeline creation
            device_id = 0 if device == "cuda" else -1
            pipeline_kwargs = dict(
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
            )
            self.pipeline = pipeline("text-generation", device=device_id, **pipeline_kwargs)

        def _use_phi_or_fallback(self):
            """Return True if Phi pipeline is available, otherwise False (use fallback)."""
            return self.is_loaded and self.pipeline is not None and not self.use_fallback

        def generate_inventory_suggestions(self, inventory_data: Dict) -> List[str]:
            """Generate AI suggestions for inventory optimization (Phi or fallback)."""
            if not self._use_phi_or_fallback():
                return get_lightweight_suggestions(inventory_data)

            try:
                total_items = len(inventory_data.get('batches', []))
                expiring_soon = len([
                    b for b in inventory_data.get('batches', [])
                    if b.get('shelf_life_prediction', {}).get('predicted_days', 10) < 3
                ])
                avg_temp = inventory_data.get('environmental_data', {}).get('temperature', 20)
                avg_humidity = inventory_data.get('environmental_data', {}).get('humidity', 60)

                prompt = f"""
    You are an AI assistant specializing in retail inventory management. Analyze the following inventory data and provide 3-5 actionable suggestions:

    Inventory Summary:
    - Total batches: {total_items}
    - Items expiring within 3 days: {expiring_soon}
    - Average storage temperature: {avg_temp}°C
    - Average storage humidity: {avg_humidity}%

    Provide specific, actionable recommendations for inventory optimization, quality management, and cost reduction.
    """

                resp = self.pipeline(prompt, max_new_tokens=300)
                text = resp[0].get('generated_text', '')
                # Simple split by lines and bullets
                suggestions = [line.strip().lstrip('-•0123456789. ').strip() for line in text.split('\n') if line.strip()]
                return suggestions[:5] if suggestions else ["Check inventory levels and optimize storage conditions."]
            except Exception as e:
                logger.error(f"Error generating Phi suggestions: {e}")
                return get_lightweight_suggestions(inventory_data)

        def generate_chat_response(self, message: str, context: Dict = None) -> str:
            """Generate chat response (Phi or fallback)."""
            if not self._use_phi_or_fallback():
                return chat_with_lightweight_ai(message, context)

            try:
                context_info = ''
                if context:
                    context_info = (
                        f"Current Business Context: Total inventory items: {len(context.get('batches', []))}. "
                        f"Items expiring soon: {len([b for b in context.get('batches', []) if b.get('shelf_life_prediction', {}).get('predicted_days', 10) < 3])}."
                    )

                prompt = f"""
    You are an AI assistant for retail inventory management. {context_info}\n
    Retailer Question: {message}
    """
                resp = self.pipeline(prompt, max_new_tokens=400)
                text = resp[0].get('generated_text', '')
                return text.strip()
            except Exception as e:
                logger.error(f"Error generating Phi chat response: {e}")
                return chat_with_lightweight_ai(message, context)

        def analyze_trends(self, analytics_data: Dict) -> Dict[str, Any]:
            """Analyze business trends using Phi or fallback."""
            if not self._use_phi_or_fallback():
                return analyze_trends_lightweight(analytics_data)

            try:
                prompt = (
                    f"Analyze these retail business metrics and provide insights:\n"
                    f"Sales Trend: {analytics_data.get('sales_trend', 'stable')}\n"
                    f"Inventory Turnover: {analytics_data.get('inventory_turnover', 'average')}\n"
                    f"Quality Issues: {analytics_data.get('quality_issues', 0)}\n"
                )
                resp = self.pipeline(prompt, max_new_tokens=250)
                text = resp[0].get('generated_text', '')
                return {"insights": text.strip(), "confidence": 0.8, "timestamp": datetime.now().isoformat()}
            except Exception as e:
                logger.error(f"Error analyzing trends with Phi: {e}")
                return analyze_trends_lightweight(analytics_data)


    # Global instance
    phi_service = PhiModelService()


    async def initialize_phi_model():
        """Initialize the Phi model service asynchronously; safe to call on startup."""
        await phi_service.load_model()


    def get_inventory_suggestions(inventory_data: Dict) -> List[str]:
        return phi_service.generate_inventory_suggestions(inventory_data)


    def chat_with_ai(message: str, context: Dict = None) -> str:
        return phi_service.generate_chat_response(message, context)


    def analyze_business_trends(analytics_data: Dict) -> Dict[str, Any]:
        return phi_service.analyze_trends(analytics_data)


# Global instance
phi_service = PhiModelService()

async def initialize_phi_model():
    """Initialize the Phi model service"""
    await phi_service.load_model()

# Utility functions for easy access
def get_inventory_suggestions(inventory_data: Dict) -> List[str]:
    """Get AI suggestions for inventory management"""
    return phi_service.generate_inventory_suggestions(inventory_data)

def chat_with_ai(message: str, context: Dict = None) -> str:
    """Chat with AI assistant"""
    return phi_service.generate_chat_response(message, context)

def analyze_business_trends(analytics_data: Dict) -> Dict[str, Any]:
    """Analyze business trends with AI"""
    return phi_service.analyze_trends(analytics_data)

def is_model_ready() -> bool:
    """Check if AI model is ready"""
    return phi_service.is_loaded