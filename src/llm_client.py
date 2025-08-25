"""Simple OpenAI LLM client for supervisor operations."""

import logging
import os
from typing import Dict, List, Optional

import openai
from openai import OpenAI

logger = logging.getLogger(__name__)


class SupervisorLLMClient:
    """Simple OpenAI client for supervisor tasks."""
    
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
    
    def analyze_problem(self, description: str, context: str, severity: str) -> Dict:
        """Analyze a problem and provide solutions."""
        system_prompt = "You are an expert supervisor helping analyze and solve problems. Provide practical, actionable solutions."
        user_prompt = f"Problem: {description}\nContext: {context}\nSeverity: {severity}\n\nPlease analyze this problem and provide 3-5 specific, actionable solutions."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            analysis_text = self.chat_completion(messages, max_tokens=500)
            solutions = self._extract_solutions(analysis_text)
            
            return {
                "analysis": f"AI Analysis for {severity} severity problem",
                "solutions": solutions,
                "estimated_time": self._estimate_time(severity),
                "ai_powered": True
            }
        except Exception as e:
            logger.error(f"Problem analysis failed: {e}")
            raise
    
    def generate_task_feedback(self, job_title: str, task_title: str, status: str, details: str) -> Dict:
        """Generate feedback for task updates."""
        system_prompt = "You are a helpful supervisor providing feedback on task progress. Be encouraging but constructive."
        user_prompt = f"Job: {job_title}\nTask: {task_title} (Status: {status})\nDetails: {details}\n\nProvide brief, encouraging feedback and next steps."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            feedback_text = self.chat_completion(messages, max_tokens=300, temperature=0.6)
            
            return {
                "message": feedback_text,
                "suggestions": ["Keep up the great work!", "Document key insights"],
                "ai_powered": True
            }
        except Exception as e:
            logger.error(f"Feedback generation failed: {e}")
            raise
    
    def breakdown_job(self, description: str) -> List[Dict]:
        """Break down a job into tasks using LLM."""
        system_prompt = """You are a project manager breaking down work into actionable tasks. 
        Create 3-5 specific, actionable tasks. For each task, provide:
        - title: Clear, concise task title
        - description: Specific what needs to be done
        - estimated_minutes: Realistic time estimate
        - priority: high, medium, or low
        
        Respond in this exact format:
        TASK 1:
        Title: [title]
        Description: [description]  
        Estimated Minutes: [number]
        Priority: [priority]
        
        TASK 2:
        [etc...]"""
        
        user_prompt = f"Break down this job into actionable tasks:\n\n{description}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.chat_completion(messages, max_tokens=800)
            return self._parse_task_breakdown(response)
        except Exception as e:
            logger.error(f"Job breakdown failed: {e}")
            raise
    
    def _extract_solutions(self, text: str) -> List[str]:
        """Extract solutions from LLM response."""
        lines = text.split('\n')
        solutions = []
        for line in lines:
            line = line.strip()
            if line and (line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '*'))):
                solutions.append(line)
        return solutions[:5]
    
    def _estimate_time(self, severity: str) -> int:
        """Estimate resolution time based on severity."""
        time_mapping = {
            "low": 30,
            "medium": 60, 
            "high": 120,
            "critical": 240
        }
        return time_mapping.get(severity.lower(), 60)
    
    def _parse_task_breakdown(self, response: str) -> List[Dict]:
        """Parse the structured task breakdown response."""
        tasks = []
        current_task = {}
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('TASK'):
                if current_task:
                    tasks.append(current_task)
                current_task = {}
            elif line.startswith('Title:'):
                current_task['title'] = line.replace('Title:', '').strip()
            elif line.startswith('Description:'):
                current_task['description'] = line.replace('Description:', '').strip()
            elif line.startswith('Estimated Minutes:'):
                try:
                    minutes = int(line.replace('Estimated Minutes:', '').strip())
                    current_task['estimated_minutes'] = minutes
                except ValueError:
                    current_task['estimated_minutes'] = 60
            elif line.startswith('Priority:'):
                priority = line.replace('Priority:', '').strip().lower()
                current_task['priority'] = priority if priority in ['high', 'medium', 'low'] else 'medium'
        
        if current_task:
            tasks.append(current_task)
        
        # Ensure we have valid tasks
        if not tasks:
            # Fallback if parsing fails
            tasks = [{
                'title': 'Complete Job',
                'description': 'Execute the required work',
                'estimated_minutes': 90,
                'priority': 'high'
            }]
        
        return tasks
