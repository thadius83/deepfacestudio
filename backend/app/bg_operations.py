"""
Background operations for DeepFace API.

This module provides functions that wrap the face processing operations
to work with the background task system.
"""
from typing import Dict, Any, List, Optional, BinaryIO, Union
from fastapi import UploadFile

from . import config
from .tasks import submit_task, get_task_result, get_task_status

# Wrapper functions for background processing

async def bg_identify_faces(
    target: Union[UploadFile, bytes], 
    threshold: Optional[float] = None
) -> str:
    """Submit face identification for background processing.
    
    Args:
        target: Image file (UploadFile) or raw bytes
        threshold: Distance threshold (0-1, lower = better match)
        
    Returns:
        Task ID for status checking
    """
    # Convert UploadFile to bytes if needed
    if isinstance(target, UploadFile):
        image_bytes = await target.read()
    else:
        image_bytes = target
        
    # Submit the task and return the task ID
    from .face_processing import identify_faces
    return submit_task(_identify_faces_sync, image_bytes, threshold)

def _identify_faces_sync(image_bytes: bytes, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    """Synchronous version of identify_faces for background processing.
    
    Args:
        image_bytes: Raw image data
        threshold: Distance threshold
        
    Returns:
        Identification results
    """
    from .utils import bgr_from_upload
    
    # Process the image directly using the utilities from face_processing.py
    try:
        # Import required functions
        from .face_processing import detect_and_identify
        import numpy as np
        
        # Convert bytes to image
        image = bgr_from_upload(image_bytes)
        
        # Call the core detection and identification function directly
        result = detect_and_identify(image, threshold)
        return result
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in _identify_faces_sync: {str(e)}")
        print(f"Error details: {error_details}")
        return [{"error": f"Error processing image: {str(e)}"}]

async def bg_analyze_face(target: Union[UploadFile, bytes]) -> str:
    """Submit face analysis for background processing.
    
    Args:
        target: Image file (UploadFile) or raw bytes
        
    Returns:
        Task ID for status checking
    """
    # Convert UploadFile to bytes if needed
    if isinstance(target, UploadFile):
        image_bytes = await target.read()
    else:
        image_bytes = target
        
    # Submit the task and return the task ID
    return submit_task(_analyze_face_sync, image_bytes)

def _analyze_face_sync(image_bytes: bytes) -> List[Dict[str, Any]]:
    """Synchronous version of analyze_face for background processing.
    
    Args:
        image_bytes: Raw image data
        
    Returns:
        Analysis results
    """
    from .utils import bgr_from_upload
    
    # Process the image directly
    try:
        from deepface import DeepFace
        from . import config
        from .utils import to_serializable
        
        # Convert bytes to image
        image = bgr_from_upload(image_bytes)
        
        # Do the face analysis directly
        result = DeepFace.analyze(
            image,
            actions=["age", "gender", "emotion", "race"],
            detector_backend=config.DETECTOR_BACKEND
        )
        
        # Ensure all outputs are JSON serializable
        return to_serializable(result)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in _analyze_face_sync: {str(e)}")
        print(f"Error details: {error_details}")
        return [{"error": f"Error analyzing face: {str(e)}"}]

async def bg_compare_faces(img1: Union[UploadFile, bytes], img2: Union[UploadFile, bytes]) -> str:
    """Submit face comparison for background processing.
    
    Args:
        img1: First image (UploadFile or bytes)
        img2: Second image (UploadFile or bytes)
        
    Returns:
        Task ID for status checking
    """
    # Convert UploadFile objects to bytes if needed
    if isinstance(img1, UploadFile):
        img1_bytes = await img1.read()
    else:
        img1_bytes = img1
        
    if isinstance(img2, UploadFile):
        img2_bytes = await img2.read()
    else:
        img2_bytes = img2
        
    # Submit the task and return the task ID
    return submit_task(_compare_faces_sync, img1_bytes, img2_bytes)

def _compare_faces_sync(img1_bytes: bytes, img2_bytes: bytes) -> Dict[str, Any]:
    """Synchronous version of compare_faces for background processing.
    
    Args:
        img1_bytes: First image raw data
        img2_bytes: Second image raw data
        
    Returns:
        Comparison results
    """
    # Create mock UploadFile-like objects
    class MockUploadFile:
        def __init__(self, data):
            self.data = data
            
        async def read(self):
            return self.data
            
        async def seek(self, position):
            pass
    
    # Call the original function synchronously
    import asyncio
    from .face_processing import compare_faces
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(compare_faces(
            MockUploadFile(img1_bytes), 
            MockUploadFile(img2_bytes)
        ))
        loop.close()
        return result
    except Exception as e:
        return {"error": f"Error comparing faces: {str(e)}"}
