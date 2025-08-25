"""Simple MCP supervisor server - minimal HTTP implementation."""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel


# Simple data models
class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"  
    COMPLETED = "completed"
    FAILED = "failed"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Task(BaseModel):
    id: str
    title: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: Priority = Priority.MEDIUM
    estimated_minutes: Optional[int] = None


class Job(BaseModel):
    id: str
    title: str
    description: str
    agent_id: str
    tasks: List[Task] = []
    created_at: datetime
    
    @property
    def progress(self) -> float:
        if not self.tasks:
            return 0.0
        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        return (completed / len(self.tasks)) * 100


# Simple in-memory storage
class SimpleStorage:
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
    
    def create_job(self, job: Job) -> Job:
        self.jobs[job.id] = job
        return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)
    
    def get_task(self, job_id: str, task_id: str) -> Optional[Task]:
        job = self.get_job(job_id)
        if not job:
            return None
        return next((t for t in job.tasks if t.id == task_id), None)


# Global storage
storage = SimpleStorage()

# FastMCP Server with HTTP transport - configured for proper host binding
mcp = FastMCP("supervisor-mcp", host="0.0.0.0", port=8000)


def break_down_job(description: str) -> List[Task]:
    """Simple rule-based task breakdown."""
    tasks = []
    
    # Basic task templates based on keywords
    if any(word in description.lower() for word in ["build", "create", "develop"]):
        tasks.extend([
            Task(id=str(uuid.uuid4()), title="Planning & Analysis", 
                 description="Analyze requirements and create plan", estimated_minutes=30),
            Task(id=str(uuid.uuid4()), title="Implementation", 
                 description="Build the core functionality", estimated_minutes=120),
            Task(id=str(uuid.uuid4()), title="Testing", 
                 description="Test and validate the solution", estimated_minutes=45),
        ])
    else:
        # Generic breakdown
        tasks.extend([
            Task(id=str(uuid.uuid4()), title="Analysis", 
                 description="Understand and break down the task", estimated_minutes=20),
            Task(id=str(uuid.uuid4()), title="Execution", 
                 description="Complete the main work", estimated_minutes=60),
            Task(id=str(uuid.uuid4()), title="Review", 
                 description="Review and finalize", estimated_minutes=15),
        ])
    
    return tasks


def get_feedback(job: Job, task: Task, status: TaskStatus, details: str) -> dict:
    """Generate simple feedback based on task update."""
    feedback = {
        "message": f"Task '{task.title}' updated to {status.value}",
        "suggestions": [],
        "next_steps": []
    }
    
    if status == TaskStatus.COMPLETED:
        feedback["message"] = f"Great! Completed '{task.title}'"
        feedback["suggestions"] = ["Document any key learnings"]
        # Find next pending task
        next_task = next((t for t in job.tasks if t.status == TaskStatus.PENDING), None)
        if next_task:
            feedback["next_steps"] = [f"Start work on '{next_task.title}'"]
    elif status == TaskStatus.IN_PROGRESS:
        feedback["suggestions"] = ["Keep focused and update progress regularly"]
        feedback["next_steps"] = ["Continue current work"]
    elif status == TaskStatus.FAILED:
        feedback["suggestions"] = ["Report the specific problem for help", "Consider breaking the task down further"]
        feedback["next_steps"] = ["Use report_problem tool for guidance"]
    
    return feedback


def solve_problem(description: str, context: str, severity: str) -> dict:
    """Generate simple problem solutions."""
    solutions = []
    
    problem_lower = description.lower()
    
    # Simple keyword matching for solutions
    if any(word in problem_lower for word in ["error", "bug", "exception"]):
        solutions.extend([
            "Check error logs for specific error details",
            "Try reproducing with minimal test case",
            "Search online for similar error messages"
        ])
    elif any(word in problem_lower for word in ["slow", "performance"]):
        solutions.extend([
            "Profile the code to identify bottlenecks",
            "Check system resources (CPU, memory)",
            "Optimize the most time-consuming operations"
        ])
    else:
        solutions.extend([
            "Break the problem into smaller parts",
            "Research similar problems online",
            "Try a different approach"
        ])
    
    return {
        "analysis": f"Analyzing {severity} severity problem",
        "solutions": solutions,
        "estimated_time": 60 if severity == "high" else 30
    }


@mcp.tool()
def start_job(job_description: str, agent_id: str, priority: str = "medium") -> dict:
    """Start a new job with task breakdown."""
    try:
        # Create job
        job_id = str(uuid.uuid4())
        title = job_description[:50] + "..." if len(job_description) > 50 else job_description
        
        job = Job(
            id=job_id,
            title=title,
            description=job_description,
            agent_id=agent_id,
            created_at=datetime.utcnow()
        )
        
        # Break down into tasks
        job.tasks = break_down_job(job_description)
        storage.create_job(job)
        
        return {
            "job_id": job_id,
            "title": job.title,
            "task_count": len(job.tasks),
            "tasks": [
                {
                    "task_id": t.id,
                    "title": t.title,
                    "description": t.description,
                    "estimated_minutes": t.estimated_minutes,
                    "status": t.status.value
                } for t in job.tasks
            ],
            "message": f"Created job with {len(job.tasks)} tasks. Start with the first task!"
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def task_update(job_id: str, task_id: str, status: str, details: str) -> dict:
    """Update task progress."""
    try:
        job = storage.get_job(job_id)
        if not job:
            return {"error": "Job not found"}
        
        task = storage.get_task(job_id, task_id)
        if not task:
            return {"error": "Task not found"}
        
        # Update task status
        task.status = TaskStatus(status)
        
        # Generate feedback
        feedback = get_feedback(job, task, task.status, details)
        
        return {
            "job_id": job_id,
            "task_id": task_id,
            "progress": f"{job.progress:.1f}%",
            "feedback": feedback
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def report_problem(job_id: str, problem_description: str, context: str, severity: str = "medium") -> dict:
    """Report a problem and get solutions."""
    try:
        job = storage.get_job(job_id)
        if not job:
            return {"error": "Job not found"}
        
        solution = solve_problem(problem_description, context, severity)
        
        return {
            "job_id": job_id,
            "problem_analysis": solution
        }
    except Exception as e:
        return {"error": str(e)}


# Run server with streamable_http transport
if __name__ == "__main__":
    mcp.run(transport="streamable-http")
