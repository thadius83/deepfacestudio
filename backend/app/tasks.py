"""
Background task processing for compute-intensive operations.
"""
import asyncio
import threading
import queue
import time
from typing import Dict, Any, Callable, Optional, List, Union
import uuid
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Task states
PENDING = "pending"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"

# Task queue and results
task_queue = queue.Queue()
task_results = {}
task_locks = {}
workers = []
MAX_WORKERS = 2  # Number of worker threads

class Task:
    """Represents a background processing task."""
    
    def __init__(self, func: Callable, *args, **kwargs):
        self.id = str(uuid.uuid4())
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.status = PENDING
        self.result = None
        self.error = None
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None
        
    def execute(self):
        """Execute the task function and capture results."""
        self.status = RUNNING
        self.started_at = time.time()
        
        try:
            # Execute the task function
            self.result = self.func(*self.args, **self.kwargs)
            self.status = COMPLETED
        except Exception as e:
            self.error = str(e)
            self.status = FAILED
            logger.error(f"Task {self.id} failed: {str(e)}")
        finally:
            self.completed_at = time.time()
            
    def to_dict(self):
        """Convert task to dictionary representation."""
        return {
            "id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "execution_time": (self.completed_at - self.started_at) if self.completed_at else None,
            "wait_time": (self.started_at - self.created_at) if self.started_at else None,
            "has_error": self.error is not None,
            "error": self.error
        }

def worker_thread():
    """Worker thread that processes tasks from the queue."""
    logger.info(f"Starting worker thread {threading.current_thread().name}")
    
    while True:
        try:
            # Get a task from the queue with a timeout
            task = task_queue.get(timeout=1.0)
            
            # Execute the task
            logger.info(f"Processing task {task.id}")
            task.execute()
            
            # Store the result
            task_results[task.id] = task
            
            # Notify the queue that the task is done
            task_queue.task_done()
            
            logger.info(f"Completed task {task.id} with status {task.status}")
        except queue.Empty:
            # Queue is empty, just continue
            continue
        except Exception as e:
            # Log any other exception
            logger.error(f"Worker thread error: {str(e)}")

def start_workers():
    """Start worker threads to process tasks."""
    global workers
    
    # Start worker threads if not already running
    if not workers:
        for i in range(MAX_WORKERS):
            t = threading.Thread(target=worker_thread, daemon=True, name=f"DeepFace-Worker-{i+1}")
            t.start()
            workers.append(t)
        logger.info(f"Started {MAX_WORKERS} worker threads")

def submit_task(func: Callable, *args, **kwargs) -> str:
    """Submit a task for background processing.
    
    Args:
        func: The function to call
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        The task ID
    """
    # Create a new task
    task = Task(func, *args, **kwargs)
    
    # Add the task to the queue
    task_queue.put(task)
    
    # Store the task in the results dict for status tracking
    task_results[task.id] = task
    
    # Create a lock for the task
    task_locks[task.id] = asyncio.Lock()
    
    # Ensure workers are running
    start_workers()
    
    return task.id

def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get the status of a task.
    
    Args:
        task_id: The ID of the task
        
    Returns:
        A dictionary with the task status information
    """
    task = task_results.get(task_id)
    
    if task is None:
        return {"id": task_id, "status": "unknown", "error": "Task not found"}
    
    return task.to_dict()

def get_task_result(task_id: str) -> Any:
    """Get the result of a completed task.
    
    Args:
        task_id: The ID of the task
        
    Returns:
        The result of the task, or None if the task is not yet completed or failed
    """
    task = task_results.get(task_id)
    
    if task is None:
        return None
    
    if task.status != COMPLETED:
        return None
    
    return task.result

def clean_old_tasks(max_age_seconds: int = 3600):
    """Clean up old tasks from the results dictionary.
    
    Args:
        max_age_seconds: Maximum age of a task in seconds (default: 1 hour)
    """
    current_time = time.time()
    
    # Find tasks to remove
    tasks_to_remove = []
    for task_id, task in task_results.items():
        if task.completed_at and (current_time - task.completed_at) > max_age_seconds:
            tasks_to_remove.append(task_id)
    
    # Remove tasks
    for task_id in tasks_to_remove:
        if task_id in task_results:
            del task_results[task_id]
        if task_id in task_locks:
            del task_locks[task_id]
    
    if tasks_to_remove:
        logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")

async def wait_for_task(task_id: str, timeout: float = 300.0) -> Dict[str, Any]:
    """Wait for a task to complete asynchronously.
    
    Args:
        task_id: The ID of the task
        timeout: Maximum time to wait in seconds
        
    Returns:
        The task status dictionary
    """
    start_time = time.time()
    
    while True:
        # Get the current task status
        task = task_results.get(task_id)
        
        if task is None:
            return {"id": task_id, "status": "unknown", "error": "Task not found"}
        
        # If the task is completed or failed, return the status
        if task.status in [COMPLETED, FAILED]:
            return task.to_dict()
        
        # Check if we've exceeded the timeout
        if (time.time() - start_time) > timeout:
            return {
                "id": task_id, 
                "status": "timeout", 
                "error": f"Timed out after {timeout} seconds"
            }
        
        # Wait a short time before checking again
        await asyncio.sleep(0.2)

# Start the worker threads when this module is imported
start_workers()
