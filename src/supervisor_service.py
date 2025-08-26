"""Supervisor service for job and task management with business logic."""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Protocol

from schemas import (
    Job, Task, TaskStatus, Priority,
    TaskBreakdownResponse, SimpleProblemAnalysis, SimpleFeedbackResponse,
    ProblemInput, ProblemAnalysisTask
)

import prompts

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
    
    def create_job(self, description: str, priority: str = "medium") -> Dict:
        """Create a new job with LLM-powered task breakdown."""
        try:
            # Create job with clean title generation
            title = self._generate_clean_title(description)
            job = Job(
                id=str(uuid.uuid4()),
                title=title,
                description=description
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
        """Update task status with simplified feedback logic."""
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
            
            # Generate simplified feedback based on status
            next_task = job.get_next_pending_task()
            next_task_title = next_task.title if next_task else None
            
            if task_status == TaskStatus.COMPLETED:
                # Use centralized completion messages
                supervisor_message = prompts.get_task_completion_message(next_task is not None)
                    
            elif task_status == TaskStatus.IN_PROGRESS and old_status == TaskStatus.PENDING:
                # Simple acknowledgment for newly started tasks
                supervisor_message = prompts.TASK_STARTED_ACKNOWLEDGMENT
                
            elif task_status == TaskStatus.IN_PROGRESS:
                # Use LLM for in-progress updates only
                feedback = self._generate_simple_feedback(task.title, details)
                supervisor_message = feedback["supervisor_message"]
                
            elif task_status == TaskStatus.FAILED:
                supervisor_message = prompts.TASK_FAILED_MESSAGE
                
            else:
                supervisor_message = prompts.TASK_UPDATED_GENERIC
            
            logger.info(f"Updated task {task_id} from {old_status} to {task_status}")
            
            return {
                "job_id": job_id,
                "task_id": task_id,
                "progress": f"{job.progress:.1f}%",
                "supervisor_message": supervisor_message,
                "next_task": next_task_title
            }
            
        except ValueError as e:
            return {"error": f"Invalid status: {status}"}
        except Exception as e:
            logger.error(f"Failed to update task: {e}")
            return {"error": f"Failed to update task: {str(e)}"}
    
    def report_problem(self, job_id: Optional[str], problem_input: ProblemInput) -> Dict:
        """Report and analyze a problem with LLM intelligence using map-reduce approach."""
        try:
            job = None
            if job_id:
                job = self.storage.get_job(job_id)
                if not job:
                    return {"error": "Job not found"}
            
            # Use map-reduce approach for problem analysis (with or without job context)
            solution = self.analyze_problem_with_map_reduce(problem_input, job)
            
            logger.info(f"Analyzed problem{' for job ' + job_id if job_id else ' without job context'}")
            
            result = {
                "problem_analysis": solution
            }
            
            # Only include job_id in response if it was provided
            if job_id:
                result["job_id"] = job_id
                
            return result
            
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
    
    def get_all_jobs(self) -> Dict:
        """Get comprehensive list of all jobs with their current status."""
        try:
            jobs = self.storage.get_all_jobs()
            
            job_summaries = []
            for job in jobs:
                job_summaries.append({
                    "job_id": job.id,
                    "title": job.title,
                    "description": job.description,
                    "progress": f"{job.progress:.1f}%",
                    "task_count": len(job.tasks),
                    "completed_tasks": len([t for t in job.tasks if t.status == TaskStatus.COMPLETED]),
                    "is_completed": job.is_completed,
                    "created_at": job.created_at.isoformat(),
                    "updated_at": job.updated_at.isoformat()
                })
            
            return {
                "jobs": job_summaries,
                "total_jobs": len(jobs),
                "message": f"Retrieved {len(jobs)} jobs successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to get all jobs: {e}")
            return {"error": f"Failed to retrieve jobs: {str(e)}"}
    
    def get_job_tasks(self, job_id: str) -> Dict:
        """Get detailed task information for a specific job."""
        try:
            job = self.storage.get_job(job_id)
            if not job:
                return {"error": "Job not found"}
            
            task_details = []
            for task in job.tasks:
                task_details.append({
                    "task_id": task.id,
                    "title": task.title,
                    "description": task.description,
                    "status": task.status.value,
                    "priority": task.priority.value,
                    "estimated_minutes": task.estimated_minutes,
                    "created_at": task.created_at.isoformat(),
                    "updated_at": task.updated_at.isoformat()
                })
            
            return {
                "job_id": job_id,
                "job_title": job.title,
                "tasks": task_details,
                "progress": f"{job.progress:.1f}%",
                "total_tasks": len(job.tasks),
                "message": f"Retrieved {len(job.tasks)} tasks for job '{job.title}'"
            }
            
        except Exception as e:
            logger.error(f"Failed to get job tasks: {e}")
            return {"error": f"Failed to retrieve job tasks: {str(e)}"}
    
    def analyze_problem_with_map_reduce(self, problem_input: ProblemInput, job: Optional[Job]) -> Dict:
        """Analyze a problem using map-reduce approach with three parallel analysis tasks."""
        
        # Map phase: Run three parallel analyses using centralized focuses
        analysis_tasks = []
        for focus in prompts.PROBLEM_ANALYSIS_FOCUSES:
            try:
                task = self._analyze_single_focus(problem_input, focus, job)
                analysis_tasks.append(task)
            except Exception as e:
                logger.error(f"Failed analysis for focus '{focus[:50]}...': {e}")
                # Continue with other analyses even if one fails
                analysis_tasks.append(ProblemAnalysisTask(
                    focus=focus[:100] + "..." if len(focus) > 100 else focus,
                    context="Analysis failed due to error",
                    observation=f"Error occurred: {str(e)}"
                ))
        
        # Reduce phase: Aggregate and synthesize findings
        return self._synthesize_analysis_results(problem_input, analysis_tasks)
    
    def _analyze_single_focus(self, problem_input: ProblemInput, focus: str, job: Optional[Job]) -> ProblemAnalysisTask:
        """Analyze the problem from a single focus perspective."""
        # Prepare context about recent tasks (if job is available)
        recent_tasks_context = ""
        if job and job.tasks:
            recent_tasks = job.tasks[-5:]  # Last 5 tasks for context
            recent_tasks_context = "Recent tasks:\n" + "\n".join([
                f"- {task.title} ({task.status.value}): {task.description}"
                for task in recent_tasks
            ])
        elif job:
            recent_tasks_context = "No tasks available in this job yet."
        else:
            recent_tasks_context = "No job context available - analyzing problem independently."
        
        # Use centralized prompt builder
        messages = prompts.build_single_focus_analysis_messages(
            problem_input.problem_description,
            problem_input.steps_taken,
            focus,
            recent_tasks_context
        )
        
        try:
            # Get a simple text response for the observation
            response = self.llm_client.chat_completion(
                messages=messages,
                max_tokens=300,
                temperature=0.4
            )
            
            return ProblemAnalysisTask(
                focus=focus,
                context=f"Problem: {problem_input.problem_description}",
                observation=response.strip()
            )
        except Exception as e:
            logger.error(f"Failed to analyze focus '{focus[:50]}...': {e}")
            raise
    
    def _synthesize_analysis_results(self, problem_input: ProblemInput, analysis_tasks: List[ProblemAnalysisTask]) -> Dict:
        """Synthesize the results from all analysis tasks into final recommendations."""
        # Prepare the synthesis input
        analyses_text = "\n\n".join([
            f"Focus: {task.focus}\nObservation: {task.observation}"
            for task in analysis_tasks
        ])
        
        # Use centralized prompt builder
        messages = prompts.build_problem_synthesis_messages(
            problem_input.problem_description,
            problem_input.steps_taken,
            analyses_text
        )
        
        try:
            # Use structured outputs for reliable final analysis
            analysis = self.llm_client.structured_completion(
                messages=messages, 
                response_model=SimpleProblemAnalysis,
                max_tokens=700,
                temperature=0.4
            )
            
            return {
                "analysis": analysis.analysis_summary,
                "solutions": [
                    {
                        "title": sol.title,
                        "description": sol.description
                    }
                    for sol in analysis.solutions
                ],
                "analysis_tasks": [
                    {
                        "focus": task.focus,
                        "observation": task.observation
                    }
                    for task in analysis_tasks
                ],
                "ai_powered": True
            }
        except Exception as e:
            logger.error(f"Problem synthesis failed: {e}")
            raise

    def _generate_simple_feedback(self, task_title: str, details: str) -> Dict:
        """Generate simple feedback for in-progress tasks using LLM."""
        # Use centralized prompt builder
        messages = prompts.build_simple_feedback_messages(task_title, details)
        
        try:
            # Use structured outputs for consistent simple feedback
            feedback = self.llm_client.structured_completion(
                messages=messages, 
                response_model=SimpleFeedbackResponse,
                max_tokens=150, 
                temperature=0.6
            )
            
            return {
                "supervisor_message": feedback.supervisor_message,
                "next_task": feedback.next_task
            }
        except Exception as e:
            logger.error(f"Simple feedback generation failed: {e}")
            # Use centralized fallback message
            return {
                "supervisor_message": prompts.FEEDBACK_FALLBACK_MESSAGE,
                "next_task": None
            }
    
    def breakdown_job(self, description: str) -> List[Dict]:
        """Break down a job into tasks using structured LLM outputs.
        
        Following Clean Code principles: single responsibility (job breakdown),
        dependency inversion (uses LLM client interface), and eliminating manual parsing.
        """
        # Use centralized prompt builder
        messages = prompts.build_task_breakdown_messages(description)
        
        try:
            # Use structured outputs instead of manual parsing
            structured_response = self.llm_client.structured_completion(
                messages=messages, 
                response_model=TaskBreakdownResponse,
                max_tokens=800,
                temperature=0.3  # Lower temperature for more consistent task breakdown
            )
            
            # Convert structured response to format expected by existing code
            return [
                {
                    'title': task.title,
                    'description': task.description,
                    'estimated_minutes': task.estimated_minutes,
                    'priority': task.priority.value
                }
                for task in structured_response.tasks
            ]
            
        except Exception as e:
            logger.error(f"Structured job breakdown failed: {e}")
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
