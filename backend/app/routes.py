"""
API route handlers for DeepFace API.
"""
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from typing import List
import os
import glob
import shutil
import pickle
from pathlib import Path

from . import config
from .reference import save_reference, rebuild_reference_db
from .face_processing import compare_faces, identify_faces, analyze_face, family_resemblance

router = APIRouter()

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
                rep_keys = list(rep_data.keys())[:5] if rep_data else []
        except Exception as e:
            rep_keys = [f"Error reading file: {str(e)}"]
    else:
        rep_keys = []
    
    return {
        "reference_path": ref_path,
        "reference_exists": os.path.exists(ref_path),
        "label_count": len(ref_labels),
        "labels": ref_labels,
        "photo_count": len(ref_photos),
        "representation_files": rep_files,
        "representation_file_count": len(rep_files),
        "representation_keys_sample": rep_keys,
        "representation_entry_count": len(rep_data) if rep_data else 0
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
async def delete_reference(label: str, file_id: str):
    """Delete a specific reference file."""
    ref_dir = config.REFERENCE_DIR / label
    
    if not os.path.exists(ref_dir):
        raise HTTPException(404, f"Label not found: {label}")
    
    # Find the file with the matching ID (which is the filename without extension)
    file_found = False
    for file in os.listdir(ref_dir):
        if file.startswith(file_id):
            file_found = True
            file_path = ref_dir / file
            try:
                os.remove(file_path)
                
                # Verify deletion
                if os.path.exists(file_path):
                    raise HTTPException(500, f"Failed to delete file (still exists): {file_path}")
                
                # Force DB rebuild after deletion
                rebuild_result = rebuild_reference_db()
                
                return {"deleted": f"{label}/{file}", "success": True, "db_rebuilt": rebuild_result}
            except Exception as e:
                raise HTTPException(500, f"Error deleting file: {str(e)}")
    
    if not file_found:
        raise HTTPException(404, f"File not found: {file_id}")

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

@router.post("/identify")
async def identify(target: UploadFile = File(...)):
    """Identify faces in an image by comparing against the reference database."""
    return await identify_faces(target)

@router.post("/analyze")
async def analyze(photo: UploadFile = File(...)):
    """Analyze face for attributes like age, gender, emotion, and race."""
    return await analyze_face(photo)

@router.post("/family-resemblance")
async def compare_family(father: UploadFile = File(...), child: UploadFile = File(...), mother: UploadFile = File(...)):
    """Compare child's face with both parents to determine resemblance."""
    return await family_resemblance(father, child, mother)
