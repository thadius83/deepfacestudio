"""
Asynchronous API utilities for DeepFace Suite.

This module provides functions for making asynchronous API requests to the backend.
Instead of waiting for long-running operations to complete, the client will receive
a task ID that can be used to check the status of the task and retrieve the results
when they are ready.
"""

import requests
import streamlit as st
import time
from typing import Dict, List, Union, Optional, Any, Tuple, Callable

# Constants
DEFAULT_POLLING_INTERVAL = 1.0  # seconds
DEFAULT_TIMEOUT = 300  # seconds (5 minutes)

def submit_async_request(
    endpoint: str,
    files: Union[Dict[str, Any], List[Tuple]],
    api_url: str,
    params: Dict[str, str] = None,
    callback: Callable = None
) -> Dict[str, Any]:
    """Submit an asynchronous request to the backend.
    
    Args:
        endpoint: API endpoint path (without the /async suffix)
        files: Dictionary or list of files to upload
        api_url: Base URL for the API
        params: Optional query parameters
        callback: Optional callback function to call with progress updates
        
    Returns:
        Dictionary with task_id and initial status
    """
    try:
        # Use the async version of the endpoint
        async_endpoint = f"{endpoint}/async"
        url = f"{api_url}/{async_endpoint}"
        
        # Log the request
        with st.expander("Async API Request Details", expanded=False):
            st.write(f"Submitting to: {url}")
            if isinstance(files, dict):
                st.write(f"Files: {list(files.keys())}")
            else:
                st.write(f"Files: {len(files)} file(s)")
            if params:
                st.write(f"Params: {params}")
        
        # Make the API request
        response = requests.post(url, files=files, params=params)
        response.raise_for_status()
        
        # Get the task ID from the response
        result = response.json()
        
        if "task_id" not in result:
            raise ValueError("Response did not contain a task_id: " + str(result))
        
        return result
        
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            st.error(f"Response Status: {e.response.status_code}")
            try:
                st.error(f"Response Text: {e.response.text}")
            except:
                pass
        return {"error": str(e)}

def check_task_status(task_id: str, api_url: str) -> Dict[str, Any]:
    """Check the status of an asynchronous task.
    
    Args:
        task_id: The ID of the task to check
        api_url: Base URL for the API
        
    Returns:
        Task status information
    """
    try:
        url = f"{api_url}/tasks/{task_id}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "error": str(e)}

def get_task_result(task_id: str, api_url: str) -> Optional[Any]:
    """Get the result of a completed task.
    
    Args:
        task_id: The ID of the task
        api_url: Base URL for the API
        
    Returns:
        The task result or None if not available
    """
    try:
        url = f"{api_url}/tasks/{task_id}/result"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting task result: {str(e)}")
        return None

def wait_for_task_completion(
    task_id: str,
    api_url: str,
    timeout: float = DEFAULT_TIMEOUT,
    polling_interval: float = DEFAULT_POLLING_INTERVAL,
    progress_bar: bool = True
) -> Optional[Any]:
    """Wait for a task to complete and return the result.
    
    Args:
        task_id: The ID of the task to wait for
        api_url: Base URL for the API
        timeout: Maximum time to wait in seconds
        polling_interval: How often to check the status in seconds
        progress_bar: Whether to show a progress bar
        
    Returns:
        The task result or None if timeout or error
    """
    start_time = time.time()
    
    # Create a progress bar if requested
    if progress_bar:
        progress = st.progress(0)
        status_text = st.empty()
    
    while True:
        # Check if we've exceeded the timeout
        elapsed = time.time() - start_time
        if elapsed > timeout:
            if progress_bar:
                status_text.error(f"Timeout waiting for task completion after {timeout} seconds")
            return None
        
        # Update the progress bar
        if progress_bar:
            # Show progress as percentage of timeout elapsed
            progress_value = min(elapsed / timeout, 0.99)
            progress.progress(progress_value)
            status_text.info(f"Processing... ({int(elapsed)}s)")
        
        # Check the status
        status = check_task_status(task_id, api_url)
        
        if status["status"] == "completed":
            # Task is complete, get the result
            if progress_bar:
                progress.progress(1.0)
                status_text.success("Processing complete!")
            
            return get_task_result(task_id, api_url)
        
        elif status["status"] == "failed":
            # Task failed
            if progress_bar:
                progress.progress(1.0)
                status_text.error(f"Processing failed: {status.get('error', 'Unknown error')}")
            
            return None
        
        # Wait before checking again
        time.sleep(polling_interval)

# Async versions of the main API functions

def identify_faces_async(group_photo, api_url: str, threshold: float = None) -> Optional[List[Dict]]:
    """Identify faces in a group photo asynchronously.
    
    Args:
        group_photo: File object with the group photo
        api_url: Base URL for the API
        threshold: Optional distance threshold (0-1, lower = more confident match)
        
    Returns:
        Task submission info with task_id
    """
    params = {}
    if threshold is not None:
        threshold = round(threshold, 2)
        params = {"threshold": str(threshold)}
    
    return submit_async_request("identify", {"target": group_photo}, api_url, params=params)

def analyze_faces_async(photo, api_url: str) -> Dict[str, Any]:
    """Analyze facial attributes asynchronously.
    
    Args:
        photo: File object with the photo
        api_url: Base URL for the API
        
    Returns:
        Task submission info with task_id
    """
    return submit_async_request("analyze", {"photo": photo}, api_url)

def compare_faces_async(img1, img2, api_url: str) -> Dict[str, Any]:
    """Compare two face photos asynchronously.
    
    Args:
        img1: First image file
        img2: Second image file
        api_url: Base URL for the API
        
    Returns:
        Task submission info with task_id
    """
    return submit_async_request("compare", {"img1": img1, "img2": img2}, api_url)
