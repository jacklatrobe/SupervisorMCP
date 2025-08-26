"""Storage classes for persistent data management with clean architecture."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Protocol

from schemas import Job, Task, ProblemSolution

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


# Problem storage protocol and implementation
class ProblemStorageProtocol(Protocol):
    """Protocol defining problem storage interface."""
    
    def save_problem(self, problem: ProblemSolution) -> ProblemSolution:
        """Save a problem-solution to storage."""
        ...
    
    def get_all_problems(self) -> List[ProblemSolution]:
        """Retrieve all problems from storage."""
        ...


class JsonLineProblemStorage:
    """JSON Lines storage implementation for problem-solution data."""
    
    def __init__(self, file_path: str = "problems.jsonl"):
        self.file_path = Path(file_path)
        self._ensure_storage_exists()
        self._problems_cache: List[ProblemSolution] = []
        self._load_data()
    
    def _ensure_storage_exists(self) -> None:
        """Ensure storage file exists."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.file_path.touch()
    
    def _load_data(self) -> None:
        """Load all problems from storage into memory cache."""
        try:
            if self.file_path.stat().st_size == 0:
                return
                
            with open(self.file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if line:
                        try:
                            problem_data = json.loads(line)
                            problem = ProblemSolution(**problem_data)
                            self._problems_cache.append(problem)
                        except (json.JSONDecodeError, ValueError) as e:
                            logger.warning(f"Failed to parse problem data: {e}")
        except Exception as e:
            logger.error(f"Failed to load problem storage: {e}")
    
    def _save_all_data(self) -> None:
        """Save all problems to storage."""
        try:
            with open(self.file_path, 'w', encoding='utf-8') as file:
                for problem in self._problems_cache:
                    problem_json = problem.model_dump_json()
                    file.write(f"{problem_json}\n")
        except Exception as e:
            logger.error(f"Failed to save problem storage: {e}")
            raise
    
    def save_problem(self, problem: ProblemSolution) -> ProblemSolution:
        """Save a problem-solution to storage."""
        # Validate problem before saving
        if not problem.problem_description or not problem.problem_description.strip():
            raise ValueError("Problem description cannot be empty")
        
        if not problem.solutions or len(problem.solutions) == 0:
            raise ValueError("Problem must have at least one solution")
        
        for solution in problem.solutions:
            if not solution.title or not solution.title.strip():
                raise ValueError("Solution title cannot be empty")
            if not solution.description or not solution.description.strip():
                raise ValueError("Solution description cannot be empty")
        
        problem.updated_at = datetime.utcnow()
        self._problems_cache.append(problem)
        
        # Save with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self._save_all_data()
                return problem
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to save problem after {max_retries} attempts: {e}")
                    raise
                else:
                    logger.warning(f"Failed to save problem (attempt {attempt + 1}): {e}")
                    # Remove from cache if save failed
                    if problem in self._problems_cache:
                        self._problems_cache.remove(problem)
    
    def get_all_problems(self) -> List[ProblemSolution]:
        """Retrieve all problems from storage."""
        return list(self._problems_cache)
