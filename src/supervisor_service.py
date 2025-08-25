"""Supervisor service for job and task management with business logic."""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Protocol

from schemas import Job, Task, TaskStatus, Priority

logger = logging.getLogger(__name__)

# Constants following clean code practices
DEFAULT_STORAGE_FILE = "supervisor_data.jsonl"


# Abstract interfaces following SOLID principles
class StorageProtocol(Protocol):
    """Protocol defining storage interface for dependency inversion."""
    
    def save_job(self, job: Job) -> Job:
        """Save a job to storage."""
        ...
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Retrieve a job from storage."""
        ...
    
    def get_all_jobs(self) -> List[Job]:
        """Retrieve all jobs from storage."""
        ...
    
    def get_task(self, job_id: str, task_id: str) -> Optional[Task]:
        """Retrieve a specific task."""
        ...
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job and all its tasks from storage."""
        ...


# Concrete implementations following clean architecture
class JsonLineStorage:
    """JSON Lines storage implementation for persistent data."""
    
    def __init__(self, file_path: str = DEFAULT_STORAGE_FILE):
        self.file_path = Path(file_path)
        self._ensure_storage_exists()
        self._jobs_cache: Dict[str, Job] = {}
        self._load_data()
    
    def _ensure_storage_exists(self) -> None:
        """Ensure storage file exists."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.file_path.touch()
    
    def _load_data(self) -> None:
        """Load all jobs from storage into memory cache."""
        try:
            if self.file_path.stat().st_size == 0:
                return
                
            with open(self.file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if line:
                        try:
                            job_data = json.loads(line)
                            job = Job(**job_data)
                            self._jobs_cache[job.id] = job
                        except (json.JSONDecodeError, ValueError) as e:
                            logger.warning(f"Failed to parse job data: {e}")
        except Exception as e:
            logger.error(f"Failed to load storage: {e}")
    
    def _save_all_data(self) -> None:
        """Save all jobs to storage."""
        try:
            with open(self.file_path, 'w', encoding='utf-8') as file:
                for job in self._jobs_cache.values():
                    job_json = job.model_dump_json()
                    file.write(f"{job_json}\n")
        except Exception as e:
            logger.error(f"Failed to save storage: {e}")
            raise
    
    def save_job(self, job: Job) -> Job:
        """Save a job to storage."""
        job.updated_at = datetime.utcnow()
        self._jobs_cache[job.id] = job
        self._save_all_data()
        return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Retrieve a job from storage."""
        return self._jobs_cache.get(job_id)
    
    def get_all_jobs(self) -> List[Job]:
        """Retrieve all jobs from storage."""
        return list(self._jobs_cache.values())
    
    def get_task(self, job_id: str, task_id: str) -> Optional[Task]:
        """Retrieve a specific task."""
        job = self.get_job(job_id)
        if not job:
            return None
        return next((task for task in job.tasks if task.id == task_id), None)
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job and all its tasks from storage."""
        if job_id in self._jobs_cache:
            del self._jobs_cache[job_id]
            self._save_all_data()
            return True
        return False


# Service composition following dependency injection
class SupervisorService:
    """Main supervisor service coordinating all components."""
    
    def __init__(self, storage: StorageProtocol, llm_client):
        self.storage = storage
        self.llm_client = llm_client
    
    def create_job(self, description: str, agent_id: str, priority: str = "medium") -> Dict:
        """Create a new job with LLM-powered task breakdown."""
        try:
            # Create job with clean title generation
            title = self._generate_clean_title(description)
            job = Job(
                id=str(uuid.uuid4()),
                title=title,
                description=description,
                agent_id=agent_id
            )
            
            # Generate tasks using LLM
            task_data = self.breakdown_job(description)
            job.tasks = self._create_tasks_from_llm_data(task_data)
            
            # Save job
            saved_job = self.storage.save_job(job)
            
            logger.info(f"Created job {saved_job.id} with {len(saved_job.tasks)} tasks")
            
            return {
                "job_id": saved_job.id,
                "title": saved_job.title,
                "task_count": len(saved_job.tasks),
                "tasks": [self._task_to_dict(task) for task in saved_job.tasks],
                "message": f"Successfully created job with {len(saved_job.tasks)} tasks. Ready to start!"
            }
            
        except Exception as e:
            logger.error(f"Failed to create job: {e}")
            return {"error": f"Failed to create job: {str(e)}"}
    
    def update_task(self, job_id: str, task_id: str, status: str, details: str) -> Dict:
        """Update task status with LLM-powered feedback."""
        try:
            job = self.storage.get_job(job_id)
            if not job:
                return {"error": "Job not found"}
            
            task = self.storage.get_task(job_id, task_id)
            if not task:
                return {"error": "Task not found"}
            
            # Update task status using clean methods
            old_status = task.status
            task_status = TaskStatus(status)
            
            if task_status == TaskStatus.COMPLETED:
                task.mark_completed()
            elif task_status == TaskStatus.IN_PROGRESS:
                task.mark_in_progress()
            elif task_status == TaskStatus.FAILED:
                task.mark_failed()
            
            # Save updated job
            self.storage.save_job(job)
            
            # Generate LLM feedback
            feedback = self.generate_task_feedback(
                job.title, task.title, task.status.value, details
            )
            
            # Add next steps
            next_task = job.get_next_pending_task()
            if next_task:
                feedback["next_steps"] = [f"Continue with: {next_task.title}"]
            else:
                feedback["next_steps"] = ["All tasks completed!"]
            
            logger.info(f"Updated task {task_id} from {old_status} to {task_status}")
            
            return {
                "job_id": job_id,
                "task_id": task_id,
                "progress": f"{job.progress:.1f}%",
                "feedback": feedback
            }
            
        except ValueError as e:
            return {"error": f"Invalid status: {status}"}
        except Exception as e:
            logger.error(f"Failed to update task: {e}")
            return {"error": f"Failed to update task: {str(e)}"}
    
    def report_problem(self, job_id: str, problem_description: str, context: str, severity: str = "medium") -> Dict:
        """Report and analyze a problem with LLM intelligence."""
        try:
            job = self.storage.get_job(job_id)
            if not job:
                return {"error": "Job not found"}
            
            # Use LLM for problem analysis
            solution = self.analyze_problem(problem_description, context, severity)
            
            logger.info(f"Analyzed problem for job {job_id}")
            
            return {
                "job_id": job_id,
                "problem_analysis": solution
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze problem: {e}")
            return {"error": f"Failed to analyze problem: {str(e)}"}
    
    def complete_task(self, job_id: str, task_id: str, completion_notes: str = "") -> Dict:
        """Complete a task and suggest next actions."""
        return self.update_task(job_id, task_id, TaskStatus.COMPLETED.value, completion_notes)
    
    def prune_job(self, job_id: str) -> Dict:
        """Delete a job and all its associated tasks."""
        try:
            # Check if job exists
            job = self.storage.get_job(job_id)
            if not job:
                return {"error": "Job not found"}
            
            # Store job info for response
            job_title = job.title
            task_count = len(job.tasks)
            
            # Delete the job and all its tasks
            success = self.storage.delete_job(job_id)
            
            if success:
                logger.info(f"Pruned job {job_id} ({job_title}) with {task_count} tasks")
                return {
                    "job_id": job_id,
                    "title": job_title,
                    "tasks_deleted": task_count,
                    "message": f"Successfully pruned job '{job_title}' and all {task_count} associated tasks",
                    "success": True
                }
            else:
                return {"error": "Failed to delete job"}
                
        except Exception as e:
            logger.error(f"Failed to prune job {job_id}: {e}")
            return {"error": f"Failed to prune job: {str(e)}"}
    
    def analyze_problem(self, description: str, context: str, severity: str) -> Dict:
        """Analyze a problem and provide solutions using LLM."""
        system_prompt = "You are an expert supervisor helping analyze and solve problems. Provide practical, actionable solutions."
        user_prompt = f"Problem: {description}\nContext: {context}\nSeverity: {severity}\n\nPlease analyze this problem and provide 3-5 specific, actionable solutions."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            analysis_text = self.llm_client.chat_completion(messages, max_tokens=500)
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
        """Generate feedback for task updates using LLM."""
        system_prompt = "You are a helpful supervisor providing feedback on task progress. Be encouraging but constructive."
        user_prompt = f"Job: {job_title}\nTask: {task_title} (Status: {status})\nDetails: {details}\n\nProvide brief, encouraging feedback and next steps."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            feedback_text = self.llm_client.chat_completion(messages, max_tokens=300, temperature=0.6)
            
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
            response = self.llm_client.chat_completion(messages, max_tokens=800)
            return self._parse_task_breakdown(response)
        except Exception as e:
            logger.error(f"Job breakdown failed: {e}")
            raise
    
    def _generate_clean_title(self, description: str) -> str:
        """Generate a clean, concise title from description."""
        title = description.strip()[:50]
        if len(description) > 50:
            title = title.rsplit(' ', 1)[0] + "..."
        return title
    
    def _task_to_dict(self, task: Task) -> Dict:
        """Convert task to dictionary representation."""
        return {
            "task_id": task.id,
            "title": task.title,
            "description": task.description,
            "status": task.status.value,
            "priority": task.priority.value,
            "estimated_minutes": task.estimated_minutes
        }
    
    def _create_tasks_from_llm_data(self, task_data: List[Dict]) -> List[Task]:
        """Create Task objects from LLM breakdown data."""
        tasks = []
        for data in task_data:
            task = Task(
                id=str(uuid.uuid4()),
                title=data.get('title', 'Untitled Task'),
                description=data.get('description', 'Task description'),
                estimated_minutes=data.get('estimated_minutes', 60),
                priority=Priority(data.get('priority', 'medium'))
            )
            tasks.append(task)
        return tasks
    
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


# Initialize services following dependency injection pattern
def create_supervisor_service():
    """Factory function to create configured supervisor service."""
    from llm_client import SupervisorLLMClient
    
    # Initialize storage
    storage = JsonLineStorage()
    
    # Initialize LLM client (required)
    llm_client = SupervisorLLMClient()
    
    # Create main service
    return SupervisorService(storage, llm_client)
