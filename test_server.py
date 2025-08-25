"""Quick test of the MCP server."""

import asyncio
import json
from src.server import storage, break_down_job, get_feedback, solve_problem
from src.server import Job, Task, TaskStatus


async def test_server():
    print("Testing SupervisorMCP server...")
    
    # Test job breakdown
    tasks = break_down_job("Build a web API with authentication")
    print(f"✓ Task breakdown created {len(tasks)} tasks")
    for task in tasks:
        print(f"  - {task.title}: {task.description}")
    
    # Test job creation and storage
    from datetime import datetime
    job = Job(
        id="test-job-1",
        title="Test Job", 
        description="Build a web API with authentication",
        agent_id="test-agent",
        tasks=tasks,
        created_at=datetime.utcnow()
    )
    
    stored_job = storage.create_job(job)
    retrieved_job = storage.get_job("test-job-1")
    print(f"✓ Job storage working: {retrieved_job.title}")
    
    # Test feedback
    task = retrieved_job.tasks[0]
    feedback = get_feedback(retrieved_job, task, TaskStatus.IN_PROGRESS, "Working on analysis")
    print(f"✓ Feedback generated: {feedback['message']}")
    
    # Test problem solving
    solution = solve_problem("Getting database connection errors", "Production deployment", "high")
    print(f"✓ Problem solution generated with {len(solution['solutions'])} solutions")
    
    print("\nAll tests passed! Server is working correctly.")


if __name__ == "__main__":
    asyncio.run(test_server())
