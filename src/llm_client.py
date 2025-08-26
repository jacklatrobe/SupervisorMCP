"""OpenAI LLM client with Structured Outputs support for supervisor operations."""

import logging
import os
from typing import Dict, List, Optional, Type, TypeVar

import openai
from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Generic type for Pydantic models
T = TypeVar('T', bound=BaseModel)


class SupervisorLLMClient:
    """OpenAI client with Structured Outputs support following SOLID principles."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the OpenAI client with structured outputs capability.
        
        Args:
            api_key: OpenAI API key, uses OPENAI_API_KEY env var if not provided
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-2024-08-06"  # Model that supports structured outputs
        logger.info("SupervisorLLM client initialized with structured outputs support")
    
    def chat_completion(self, messages: List[Dict], max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Get a traditional chat completion from OpenAI (legacy method).
        
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
    
    def structured_completion(
        self, 
        messages: List[Dict], 
        response_model: Type[T], 
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> T:
        """Get a structured completion using Pydantic models.
        
        Following Clean Code principles with clear separation of concerns.
        This method ensures type-safe, validated responses that eliminate parsing errors.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            response_model: Pydantic model class for response structure
            max_tokens: Maximum tokens in response
            temperature: Response creativity (0.0-1.0)
            
        Returns:
            Parsed and validated response as Pydantic model instance
            
        Raises:
            Exception: If API call fails or response cannot be parsed
        """
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=response_model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Handle refusals explicitly (SOLID: explicit error handling)
            if response.choices[0].message.refusal:
                logger.warning(f"Model refused to respond: {response.choices[0].message.refusal}")
                raise ValueError(f"Model refusal: {response.choices[0].message.refusal}")
            
            parsed_response = response.choices[0].message.parsed
            if parsed_response is None:
                raise ValueError("Failed to parse response into structured format")
                
            logger.debug(f"Successfully parsed structured response: {type(parsed_response).__name__}")
            return parsed_response
            
        except Exception as e:
            logger.error(f"Structured completion failed: {e}")
            raise
