"""Pydantic data models for the supervisor system."""

from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


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
