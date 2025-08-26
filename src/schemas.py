"""Pydantic data models for the supervisor system."""

from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


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


# Structured Output Schemas for LLM Responses
# Following Clean Code principles with clear, focused schemas

class TaskBreakdownItem(BaseModel):
    """Schema for individual task in LLM breakdown response."""
    title: str = Field(..., description="Clear, concise task title")
    description: str = Field(..., description="Specific description of what needs to be done")
    estimated_minutes: int = Field(..., ge=1, le=600, description="Realistic time estimate in minutes")
    priority: Priority = Field(..., description="Task priority level")


class TaskBreakdownResponse(BaseModel):
    """Schema for LLM task breakdown response."""
    tasks: List[TaskBreakdownItem] = Field(..., min_items=1, max_items=10, description="List of actionable tasks")


class ProblemInput(BaseModel):
    """Schema for problem input data."""
    problem_description: str = Field(..., description="Description of the problem")
    steps_taken: List[str] = Field(..., description="List of steps taken so far")


class ProblemAnalysisTask(BaseModel):
    """Schema for individual problem analysis task."""
    focus: str = Field(..., description="Specific focus area for analysis")
    context: str = Field(..., description="Context information for the analysis")
    observation: str = Field(..., description="Observation or finding from the analysis")


class Solution(BaseModel):
    """Schema for individual problem solution."""
    title: str = Field(..., description="Brief solution title")
    description: str = Field(..., description="Detailed solution steps")


class SimpleProblemAnalysis(BaseModel):
    """Simplified schema for problem analysis results."""
    analysis_summary: str = Field(..., description="Brief analysis of the problem")
    solutions: List[Solution] = Field(..., min_items=1, max_items=5, description="Actionable solutions")


class SimpleFeedbackResponse(BaseModel):
    """Simplified schema for task feedback - only used for in-progress updates."""
    supervisor_message: str = Field(..., description="Brief guidance message from supervisor")
    next_task: Optional[str] = Field(default=None, description="Title of next task if available")
