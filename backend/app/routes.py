"""
API route handlers for DeepFace API.
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Optional, Dict, Any
import os
import glob
import shutil
import pickle
from pathlib import Path
import time

from . import config
from .reference import save_reference, rebuild_reference_db
from .face_processing import compare_faces, identify_faces, analyze_face, family_resemblance
from .tasks import get_task_status, get_task_result, clean_old_tasks
from .bg_operations import bg_identify_faces, bg_analyze_face, bg_compare_faces

router = APIRouter()

# ---------- Task management routes ----------
@router.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get the status of a background task."""
    # Clean up old tasks periodically
    clean_old_tasks()
    
    # Get the task status
    status = get_task_status(task_id)
    
    if status["status"] == "unknown":
        raise HTTPException(404, f"Task not found: {task_id}")
    
    return status

@router.get("/tasks/{task_id}/result")
async def get_task_result_endpoint(task_id: str):
    """Get the result of a completed background task."""
    # Clean up old tasks periodically
    clean_old_tasks()
    
    # Get the task status
    status = get_task_status(task_id)
    
    if status["status"] == "unknown":
        raise HTTPException(404, f"Task not found: {task_id}")
    
    if status["status"] == "pending" or status["status"] == "running":
        return {"status": status["status"], "message": "Task is still in progress"}
    
    if status["status"] == "failed":
        raise HTTPException(500, f"Task failed: {status.get('error', 'Unknown error')}")
    
    # Get the task result
    result = get_task_result(task_id)
    
    if result is None:
        raise HTTPException(404, "Task result not found")
    
    return result

# ---------- Reference image routes ----------
@router.get("/reference/image/{label}/{file_id}")
async def get_reference_image(label: str, file_id: str):
    """Serve a reference image file."""
    ref_dir = config.REFERENCE_DIR / label
    
    if not os.path.exists(ref_dir):
        raise HTTPException(404, f"Label not found: {label}")
    
    # Find the file with the matching ID (which is the filename without extension)
    for file in os.listdir(ref_dir):
        if file.startswith(file_id):
            file_path = ref_dir / file
            if os.path.exists(file_path):
                return FileResponse(file_path, media_type="image/jpeg")
    
    raise HTTPException(404, f"File not found: {file_id}")

@router.get("/reference/status")
async def reference_status():
    """Get the status of the reference database, including representation file info."""
    import os
    import glob
    from pathlib import Path
    import pickle
    
    ref_path = str(config.REFERENCE_DIR)
    
    # Find representation files
    rep_files = []
    for pattern in ["representations_*.pkl", "*.pkl"]:
        rep_files.extend(glob.glob(os.path.join(ref_path, pattern)))
    
    # Count reference photos
    ref_labels = []
    ref_photos = []
    
    if os.path.exists(ref_path):
        for item in os.listdir(ref_path):
            item_path = os.path.join(ref_path, item)
            if os.path.isdir(item_path) and not item.startswith('.') and not item.startswith('ds_model'):
                ref_labels.append(item)
                for ext in config.ALLOWED_EXT:
                    photos = glob.glob(os.path.join(item_path, f"*{ext}"))
                    ref_photos.extend(photos)
    
    # Try to read the representation file
    rep_data = {}
    if rep_files:
        try:
            with open(rep_files[0], 'rb') as f:
                rep_data = pickle.load(f)
                # Get a sample of representation keys for debugging
                if isinstance(rep_data, dict):
                    rep_keys = list(rep_data.keys())[:5] if rep_data else []
                    rep_count = len(rep_data)
                elif isinstance(rep_data, list):
                    # Handle list-type representation data
                    rep_keys = [f"List data with {len(rep_data)} items"]
                    rep_count = len(rep_data)
                else:
                    rep_keys = [f"Unknown data type: {type(rep_data)}"]
                    rep_count = 0
        except Exception as e:
            rep_keys = [f"Error reading file: {str(e)}"]
            rep_count = 0
    else:
        rep_keys = []
        rep_count = 0
    
    return {
        "reference_path": ref_path,
        "reference_exists": os.path.exists(ref_path),
        "label_count": len(ref_labels),
        "labels": ref_labels,
        "photo_count": len(ref_photos),
        "representation_files": rep_files,
        "representation_file_count": len(rep_files),
        "representation_keys_sample": rep_keys,
        "representation_entry_count": rep_count
    }

@router.get("/reference")
async def list_references():
    """List all references organized by label."""
    ref_path = config.REFERENCE_DIR
    result = {"labels": {}}
    
    if not os.path.exists(ref_path):
        return {"error": f"Reference directory does not exist: {ref_path}"}
    
    # List all subdirectories (labels) and their files
    for item in os.listdir(ref_path):
        # Skip hidden files and the models file
        if item.startswith('.') or item.startswith('ds_model'):
            continue
        
        item_path = Path(ref_path) / item
        if os.path.isdir(item_path):
            files = []
            for file in os.listdir(item_path):
                if file.endswith(config.ALLOWED_EXT):
                    file_path = f"{item}/{file}"
                    file_id = file.split('.')[0]  # Extract UUID
                    files.append({
                        "id": file_id,
                        "path": file_path,
                        "filename": file
                    })
            
            result["labels"][item] = {
                "count": len(files),
                "files": files
            }
    
    return result

@router.delete("/reference/{label}/{file_id}")
async def delete_reference(label: str, file_id: str, skip_rebuild: bool = False):
    """Delete a specific reference file.
    
    Args:
        label: The label (person name) to delete from
        file_id: The specific file ID to delete
        skip_rebuild: If True, skip rebuilding the database (for batch operations)
    """
    ref_dir = config.REFERENCE_DIR / label
    
    if not os.path.exists(ref_dir):
        raise HTTPException(404, f"Label not found: {label}")
    
    # Find the file with the matching ID (which is the filename without extension)
    file_found = False
    deleted_file = None
    
    for file in os.listdir(ref_dir):
        if file.startswith(file_id):
            file_found = True
            file_path = ref_dir / file
            deleted_file = file
            try:
                os.remove(file_path)
                
                # Verify deletion
                if os.path.exists(file_path):
                    raise HTTPException(500, f"Failed to delete file (still exists): {file_path}")
                
                # Only rebuild the database if not skipping
                rebuild_result = None
                if not skip_rebuild:
                    rebuild_result = rebuild_reference_db()
                
                return {"deleted": f"{label}/{file}", "success": True, "db_rebuilt": rebuild_result}
            except Exception as e:
                raise HTTPException(500, f"Error deleting file: {str(e)}")
    
    if not file_found:
        raise HTTPException(404, f"File not found: {file_id}")

@router.post("/reference/{label}/batch-delete")
async def batch_delete_references(label: str, file_ids: List[str]):
    """Delete multiple reference files in a batch operation.
    
    Args:
        label: The label (person name) to delete from
        file_ids: List of file IDs to delete
    """
    ref_dir = config.REFERENCE_DIR / label
    
    if not os.path.exists(ref_dir):
        raise HTTPException(404, f"Label not found: {label}")
    
    results = {
        "success": True,
        "deleted": [],
        "errors": [],
        "total_requested": len(file_ids),
        "total_deleted": 0
    }
    
    # Delete each file, but skip database rebuilding until the end
    for file_id in file_ids:
        try:
            # Use the existing delete_reference function but skip rebuilds
            await delete_reference(label, file_id, skip_rebuild=True)
            results["deleted"].append(file_id)
            results["total_deleted"] += 1
        except HTTPException as e:
            results["errors"].append({"file_id": file_id, "error": str(e.detail)})
            results["success"] = False
        except Exception as e:
            results["errors"].append({"file_id": file_id, "error": str(e)})
            results["success"] = False
    
    # Only rebuild the database once after all deletions
    if results["total_deleted"] > 0:
        rebuild_result = rebuild_reference_db()
        results["db_rebuilt"] = rebuild_result
    
    return results

@router.post("/reference/rebuild")
async def force_rebuild_db():
    """Force a complete rebuild of the reference database."""
    result = rebuild_reference_db(force=True)
    return {"success": result}

@router.delete("/reference/all")
async def delete_all_references():
    """Delete all reference data and reset the database."""
    try:
        # Check if the reference directory exists
        if not os.path.exists(config.REFERENCE_DIR):
            return {"success": False, "error": "Reference directory does not exist"}
            
        # Get all labels (directories that don't start with . or ds_model)
        labels = []
        for item in os.listdir(config.REFERENCE_DIR):
            item_path = os.path.join(config.REFERENCE_DIR, item)
            if os.path.isdir(item_path) and not item.startswith('.') and not item.startswith('ds_model'):
                labels.append(item)
        
        # Enhanced deletion
        for label in labels:
            label_dir = os.path.join(config.REFERENCE_DIR, label)
            if os.path.exists(label_dir) and os.path.isdir(label_dir):
                # Try to delete each file individually first
                for file in os.listdir(label_dir):
                    file_path = os.path.join(label_dir, file)
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass
            
                # Then try to remove the directory
                try:
                    shutil.rmtree(label_dir)
                    
                    # If directory still exists, try fallback method
                    if os.path.exists(label_dir):
                        try:
                            os.rmdir(label_dir)
                        except Exception:
                            pass
                except Exception:
                    pass
        
        # Find and delete any representation files
        pkl_files = []
        for pattern in ["representations_*.pkl", "*.pkl"]:
            pattern_path = os.path.join(str(config.REFERENCE_DIR), pattern)
            pkl_files.extend(glob.glob(pattern_path))
        
        # Delete each PKL file
        for rep_file in pkl_files:
            try:
                os.remove(rep_file)
            except Exception:
                pass
        
        # Return success response
        return {
            "success": True, 
            "deleted_labels": labels,
            "deleted_pkl_files": len(pkl_files),
            "message": "Database has been completely reset"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.delete("/reference/{label}")
async def delete_label(label: str):
    """Delete an entire label and all its references."""
    ref_dir = config.REFERENCE_DIR / label
    
    if not os.path.exists(ref_dir):
        raise HTTPException(404, f"Label not found: {label}")
    
    try:
        # Delete the entire directory
        shutil.rmtree(ref_dir)
        
        # Verify directory was deleted
        if os.path.exists(ref_dir):
            raise HTTPException(500, f"Failed to delete directory: {ref_dir}")
        
        # Force DB rebuild after deletion
        rebuild_result = rebuild_reference_db()
        
        # Return success response with details
        return {
            "deleted": label, 
            "success": True, 
            "db_rebuilt": rebuild_result,
            "message": f"Successfully deleted label '{label}' and all its photos"
        }
    except Exception as e:
        raise HTTPException(500, f"Error deleting label: {str(e)}")

@router.post("/reference/{label}")
async def add_reference(label: str, files: List[UploadFile] = File(...)):
    """Add reference images under a specific label."""
    saved = [save_reference(label, await f.read()) for f in files]
    
    # After saving references, trigger a DB rebuild
    db_rebuilt = rebuild_reference_db()
    return {"stored": saved, "db_rebuilt": db_rebuilt}

# ---------- Face processing routes ----------
@router.post("/compare")
async def compare(img1: UploadFile = File(...), img2: UploadFile = File(...)):
    """Compare two faces to verify if they are the same person."""
    return await compare_faces(img1, img2)

@router.post("/compare/async")
async def compare_async(img1: UploadFile = File(...), img2: UploadFile = File(...)):
    """Compare two faces asynchronously. Returns a task ID to check status."""
    # Read the file content first
    img1_bytes = await img1.read()
    img2_bytes = await img2.read()
    
    # Submit the task with just the bytes
    task_id = await bg_compare_faces(img1_bytes, img2_bytes)
    
    # Return the task ID for status checking
    return {"task_id": task_id, "status": "pending"}

@router.post("/identify")
async def identify(
    target: UploadFile = File(...), 
    threshold: Optional[float] = Query(None, description="Match threshold (0-1, lower=better match)")
):
    """Identify faces in an image by comparing against the reference database.
    
    Args:
        target: The uploaded image file containing one or more faces
        threshold: Optional distance threshold (0-1, lower = more confident match)
               Default is None, which will use the value from config (0.30)
    """
    # Round the threshold to 2 decimal places to avoid floating point precision issues
    if threshold is not None:
        try:
            threshold = float(threshold)
            threshold = round(threshold, 2)
            print(f"Using threshold: {threshold}")
        except (ValueError, TypeError):
            print(f"Invalid threshold value: {threshold}, using default")
            threshold = None
    
    return await identify_faces(target, threshold)

@router.post("/identify/async")
async def identify_async(
    target: UploadFile = File(...), 
    threshold: Optional[float] = Query(None, description="Match threshold (0-1, lower=better match)")
):
    """Identify faces asynchronously. Returns a task ID to check status."""
    # Validate and normalize threshold
    if threshold is not None:
        try:
            threshold = float(threshold)
            threshold = round(threshold, 2)
        except (ValueError, TypeError):
            threshold = None
    
    # Read the file content first
    image_bytes = await target.read()
    
    # Submit the task with just the bytes and threshold
    task_id = await bg_identify_faces(image_bytes, threshold)
    
    # Return the task ID for status checking
    return {"task_id": task_id, "status": "pending"}

@router.post("/analyze")
async def analyze(photo: UploadFile = File(...)):
    """Analyze face for attributes like age, gender, emotion, and race."""
    return await analyze_face(photo)

@router.post("/analyze/async")
async def analyze_async(photo: UploadFile = File(...)):
    """Analyze face attributes asynchronously. Returns a task ID to check status."""
    # Read the file content first
    image_bytes = await photo.read()
    
    # Submit the task with just the bytes
    task_id = await bg_analyze_face(image_bytes)
    
    # Return the task ID for status checking
    return {"task_id": task_id, "status": "pending"}

@router.post("/family-resemblance")
async def compare_family(father: UploadFile = File(...), child: UploadFile = File(...), mother: UploadFile = File(...)):
    """Compare child's face with both parents to determine resemblance."""
    return await family_resemblance(father, child, mother)

# ---------- Debug routes ----------
@router.get("/debug/ping")
def ping():
    """Simple debug endpoint to test connectivity."""
    print("PING endpoint called")
    return {"status": "ok", "message": "API server is running"}
