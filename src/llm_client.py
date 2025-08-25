"""Simple OpenAI LLM client for supervisor operations."""

import logging
import os
from typing import Dict, List, Optional

import openai
from openai import OpenAI

logger = logging.getLogger(__name__)


class SupervisorLLMClient:
    """Simple OpenAI client for LLM chat completions only."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key, uses OPENAI_API_KEY env var if not provided
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"
        logger.info("SupervisorLLM client initialized")
    
    def chat_completion(self, messages: List[Dict], max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Get a chat completion from OpenAI.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens in response
            temperature: Response creativity (0.0-1.0)
            
        Returns:
            The response content as a string
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise
