"""Professional MCP supervisor server with clean architecture."""

import json
import logging
from datetime import datetime
from typing import Dict

from mcp.server.fastmcp import FastMCP
from schemas import TaskStatus
from supervisor_service import create_supervisor_service

# Configure logging for MCP server (stderr only, never stdout)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # This defaults to stderr
)
logger = logging.getLogger(__name__)

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


@mcp.tool()
def prune_job(job_id: str) -> dict:
    """Delete a job and all its associated tasks.
    
    Args:
        job_id: Unique identifier for the job to prune
    
    Returns:
        Dictionary containing deletion confirmation and details
    """
    return supervisor_service.prune_job(job_id)


# MCP Resources Implementation
@mcp.resource("jobs://all")
def get_all_jobs_resource() -> str:
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
def get_job_tasks_resource(job_id: str) -> str:
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
