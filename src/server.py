"""Professional MCP supervisor server with clean architecture."""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Protocol
from enum import Enum

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from llm_client import SupervisorLLMClient


# Configure logging for MCP server (stderr only, never stdout)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # This defaults to stderr
)
logger = logging.getLogger(__name__)

# Constants following clean code practices
DEFAULT_STORAGE_FILE = "supervisor_data.jsonl"

# Data Models with improved validation
class TaskStatus(str, Enum):
    """Task status enumeration with clear states."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"  
    COMPLETED = "completed"
    FAILED = "failed"


class Priority(str, Enum):
    """Priority levels for tasks and jobs."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Task(BaseModel):
    """Task model with validation and business rules."""
    id: str = Field(..., description="Unique task identifier")
    title: str = Field(..., min_length=1, description="Task title")
    description: str = Field(..., min_length=1, description="Task description")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current task status")
    priority: Priority = Field(default=Priority.MEDIUM, description="Task priority")
    estimated_minutes: Optional[int] = Field(default=None, ge=1, description="Estimated time in minutes")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    def mark_completed(self) -> None:
        """Mark task as completed with timestamp update."""
        self.status = TaskStatus.COMPLETED
        self.updated_at = datetime.utcnow()
    
    def mark_in_progress(self) -> None:
        """Mark task as in progress with timestamp update."""
        self.status = TaskStatus.IN_PROGRESS
        self.updated_at = datetime.utcnow()
    
    def mark_failed(self) -> None:
        """Mark task as failed with timestamp update."""
        self.status = TaskStatus.FAILED
        self.updated_at = datetime.utcnow()


class Job(BaseModel):
    """Job model representing a collection of tasks."""
    id: str = Field(..., description="Unique job identifier")
    title: str = Field(..., min_length=1, description="Job title")
    description: str = Field(..., min_length=1, description="Job description")
    agent_id: str = Field(..., min_length=1, description="Agent responsible for job")
    tasks: List[Task] = Field(default_factory=list, description="Job tasks")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    @property
    def progress(self) -> float:
        """Calculate job completion percentage."""
        if not self.tasks:
            return 0.0
        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        return (completed / len(self.tasks)) * 100
    
    @property
    def is_completed(self) -> bool:
        """Check if all tasks are completed."""
        return all(task.status == TaskStatus.COMPLETED for task in self.tasks)
    
    def get_next_pending_task(self) -> Optional[Task]:
        """Get the next pending task in the job."""
        return next((task for task in self.tasks if task.status == TaskStatus.PENDING), None)


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


# Service composition following dependency injection
class SupervisorService:
    """Main supervisor service coordinating all components."""
    
    def __init__(self, storage: StorageProtocol, llm_client: SupervisorLLMClient):
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
            task_data = self.llm_client.breakdown_job(description)
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
            feedback = self.llm_client.generate_task_feedback(
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
            solution = self.llm_client.analyze_problem(problem_description, context, severity)
            
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
def create_supervisor_service() -> SupervisorService:
    """Factory function to create configured supervisor service."""
    # Initialize storage
    storage = JsonLineStorage()
    
    # Initialize LLM client (required)
    llm_client = SupervisorLLMClient()
    
    # Create main service
    return SupervisorService(storage, llm_client)


# Initialize services
supervisor_service = create_supervisor_service()

# Initialize FastMCP server with proper configuration
mcp = FastMCP("supervisor-mcp", host="0.0.0.0", port=8000)

# MCP Tools Implementation
@mcp.tool()
def start_job(job_description: str, agent_id: str, priority: str = "medium") -> dict:
    """Start a new job with intelligent task breakdown.
    
    Args:
        job_description: Detailed description of the job to be completed
        agent_id: Identifier for the agent responsible for this job
        priority: Job priority level (low, medium, high, critical)
    
    Returns:
        Dictionary containing job details, tasks, and success message
    """
    return supervisor_service.create_job(job_description, agent_id, priority)


@mcp.tool()
def update_task(job_id: str, task_id: str, status: str, details: str) -> dict:
    """Update task progress with intelligent feedback.
    
    Args:
        job_id: Unique identifier for the job
        task_id: Unique identifier for the task
        status: New task status (pending, in_progress, completed, failed)
        details: Additional details about the task update
    
    Returns:
        Dictionary containing update confirmation and intelligent feedback
    """
    return supervisor_service.update_task(job_id, task_id, status, details)


@mcp.tool()
def complete_task(job_id: str, task_id: str, completion_notes: str = "") -> dict:
    """Mark a task as completed and get next task recommendations.
    
    Args:
        job_id: Unique identifier for the job
        task_id: Unique identifier for the task
        completion_notes: Optional notes about task completion
    
    Returns:
        Dictionary containing completion confirmation and next steps
    """
    return supervisor_service.complete_task(job_id, task_id, completion_notes)


@mcp.tool()
def report_problem(job_id: str, problem_description: str, context: str, severity: str = "medium") -> dict:
    """Report a problem and receive intelligent troubleshooting advice.
    
    Args:
        job_id: Unique identifier for the job where problem occurred
        problem_description: Detailed description of the problem
        context: Additional context about when/how the problem occurred
        severity: Problem severity level (low, medium, high, critical)
    
    Returns:
        Dictionary containing problem analysis and actionable solutions
    """
    return supervisor_service.report_problem(job_id, problem_description, context, severity)


@mcp.tool()
def get_all_jobs() -> dict:
    """Get comprehensive list of all jobs with their current status.
    
    Returns:
        Dictionary containing all jobs with tasks and progress information
    """
    try:
        all_jobs = supervisor_service.storage.get_all_jobs()
        jobs_data = []
        
        for job in all_jobs:
            job_data = {
                "job_id": job.id,
                "title": job.title,
                "description": job.description,
                "agent_id": job.agent_id,
                "progress": f"{job.progress:.1f}%",
                "is_completed": job.is_completed,
                "task_count": len(job.tasks),
                "created_at": job.created_at.isoformat(),
                "updated_at": job.updated_at.isoformat(),
                "tasks_summary": {
                    "pending": sum(1 for t in job.tasks if t.status == TaskStatus.PENDING),
                    "in_progress": sum(1 for t in job.tasks if t.status == TaskStatus.IN_PROGRESS),
                    "completed": sum(1 for t in job.tasks if t.status == TaskStatus.COMPLETED),
                    "failed": sum(1 for t in job.tasks if t.status == TaskStatus.FAILED)
                }
            }
            jobs_data.append(job_data)
        
        return {
            "total_jobs": len(jobs_data),
            "jobs": jobs_data,
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve all jobs: {e}")
        return {"error": f"Failed to retrieve jobs: {str(e)}"}


@mcp.tool()
def get_job_tasks(job_id: str) -> dict:
    """Get detailed task information for a specific job.
    
    Args:
        job_id: Unique identifier for the job
    
    Returns:
        Dictionary containing all tasks for the specified job
    """
    try:
        job = supervisor_service.storage.get_job(job_id)
        if not job:
            return {"error": "Job not found"}
        
        tasks_data = []
        for task in job.tasks:
            task_data = {
                "task_id": task.id,
                "title": task.title,
                "description": task.description,
                "status": task.status.value,
                "priority": task.priority.value,
                "estimated_minutes": task.estimated_minutes,
                "created_at": task.created_at.isoformat(),
                "updated_at": task.updated_at.isoformat()
            }
            tasks_data.append(task_data)
        
        # Find next pending task
        next_task = job.get_next_pending_task()
        
        return {
            "job_id": job_id,
            "job_title": job.title,
            "progress": f"{job.progress:.1f}%",
            "total_tasks": len(tasks_data),
            "next_pending_task": next_task.title if next_task else None,
            "tasks": tasks_data,
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve tasks for job {job_id}: {e}")
        return {"error": f"Failed to retrieve tasks: {str(e)}"}


# MCP Resources Implementation
@mcp.resource("jobs://all")
def get_all_jobs() -> str:
    """Get comprehensive list of all jobs with their current status.
    
    Returns:
        JSON string containing all jobs with tasks and progress information
    """
    try:
        all_jobs = supervisor_service.storage.get_all_jobs()
        jobs_data = []
        
        for job in all_jobs:
            job_data = {
                "job_id": job.id,
                "title": job.title,
                "description": job.description,
                "agent_id": job.agent_id,
                "progress": f"{job.progress:.1f}%",
                "is_completed": job.is_completed,
                "task_count": len(job.tasks),
                "created_at": job.created_at.isoformat(),
                "updated_at": job.updated_at.isoformat(),
                "tasks_summary": {
                    "pending": sum(1 for t in job.tasks if t.status == TaskStatus.PENDING),
                    "in_progress": sum(1 for t in job.tasks if t.status == TaskStatus.IN_PROGRESS),
                    "completed": sum(1 for t in job.tasks if t.status == TaskStatus.COMPLETED),
                    "failed": sum(1 for t in job.tasks if t.status == TaskStatus.FAILED)
                }
            }
            jobs_data.append(job_data)
        
        return json.dumps({
            "total_jobs": len(jobs_data),
            "jobs": jobs_data,
            "retrieved_at": datetime.utcnow().isoformat()
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to retrieve all jobs: {e}")
        return json.dumps({"error": f"Failed to retrieve jobs: {str(e)}"})


@mcp.resource("jobs://{job_id}/tasks")
def get_job_tasks(job_id: str) -> str:
    """Get detailed task information for a specific job.
    
    Args:
        job_id: Unique identifier for the job
    
    Returns:
        JSON string containing all tasks for the specified job
    """
    try:
        job = supervisor_service.storage.get_job(job_id)
        if not job:
            return json.dumps({"error": "Job not found"})
        
        tasks_data = []
        for task in job.tasks:
            task_data = {
                "task_id": task.id,
                "title": task.title,
                "description": task.description,
                "status": task.status.value,
                "priority": task.priority.value,
                "estimated_minutes": task.estimated_minutes,
                "created_at": task.created_at.isoformat(),
                "updated_at": task.updated_at.isoformat()
            }
            tasks_data.append(task_data)
        
        # Find next pending task
        next_task = job.get_next_pending_task()
        
        return json.dumps({
            "job_id": job_id,
            "job_title": job.title,
            "progress": f"{job.progress:.1f}%",
            "total_tasks": len(tasks_data),
            "next_pending_task": next_task.title if next_task else None,
            "tasks": tasks_data,
            "retrieved_at": datetime.utcnow().isoformat()
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to retrieve tasks for job {job_id}: {e}")
        return json.dumps({"error": f"Failed to retrieve tasks: {str(e)}"})


# Run server with proper transport configuration
if __name__ == "__main__":
    logger.info("Starting SupervisorMCP server...")
    logger.info(f"Storage file: {supervisor_service.storage.file_path}")
    logger.info("LLM Client: Enabled")
    
    mcp.run(transport="streamable-http")
