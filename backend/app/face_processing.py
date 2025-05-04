"""
Face processing operations (identify, compare, analyze) for DeepFace API.
"""
from fastapi import HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
import os
import numpy as np
import cv2
import glob
from deepface import DeepFace
from . import config
from .utils import bgr_from_upload, to_serializable
from .reference import rebuild_reference_db

async def compare_faces(img1: UploadFile, img2: UploadFile):
    """Compare two faces to determine if they are the same person."""
    try:
        # First try with enforce_detection=True
        bgr1, bgr2 = bgr_from_upload(await img1.read()), bgr_from_upload(await img2.read())
        result = DeepFace.verify(
            bgr1, bgr2,
            model_name=config.MODEL_NAME,
            detector_backend=config.DETECTOR_BACKEND,
            enforce_detection=True,
            distance_metric="cosine"
        )
        return JSONResponse(result)
    except ValueError as e:
        # If face detection fails, retry with enforce_detection=False
        if "Face could not be detected" in str(e):
            bgr1, bgr2 = bgr_from_upload(await img1.seek(0) or await img1.read()), bgr_from_upload(await img2.seek(0) or await img2.read())
            result = DeepFace.verify(
                bgr1, bgr2,
                model_name=config.MODEL_NAME,
                detector_backend=config.DETECTOR_BACKEND,
                enforce_detection=False,  # Don't enforce face detection
                distance_metric="cosine"
            )
            # Add a flag to indicate we had to disable face detection
            result["face_detection_enforced"] = False
            return JSONResponse(result)
        raise  # Re-raise if it's another kind of error

async def family_resemblance(father: UploadFile, child: UploadFile, mother: UploadFile):
    """Compare child's face with both parents to determine resemblance."""
    try:
        # Load images
        father_img = bgr_from_upload(await father.read())
        child_img = bgr_from_upload(await child.read())
        mother_img = bgr_from_upload(await mother.read())
        
        # Compare father and child
        paternal_result = DeepFace.verify(
            father_img, child_img,
            model_name=config.MODEL_NAME,
            detector_backend=config.DETECTOR_BACKEND,
            enforce_detection=True,
            distance_metric="euclidean"  # Using euclidean as in the notebook example
        )
        
        # Compare mother and child
        maternal_result = DeepFace.verify(
            mother_img, child_img,
            model_name=config.MODEL_NAME,
            detector_backend=config.DETECTOR_BACKEND,
            enforce_detection=True,
            distance_metric="euclidean"
        )
        
        # Determine resemblance
        if paternal_result["distance"] < maternal_result["distance"]:
            resemblance = "father"
        else:
            resemblance = "mother"
            
        # Construct result
        result = {
            "paternal_result": paternal_result,
            "maternal_result": maternal_result,
            "resemblance": resemblance
        }
        
        return JSONResponse(result)
    except ValueError as e:
        # If face detection fails, retry with enforce_detection=False
        if "Face could not be detected" in str(e):
            try:
                # Reset file pointers
                await father.seek(0)
                await child.seek(0)
                await mother.seek(0)
                
                # Load images again
                father_img = bgr_from_upload(await father.read())
                child_img = bgr_from_upload(await child.read())
                mother_img = bgr_from_upload(await mother.read())
                
                # Compare father and child
                paternal_result = DeepFace.verify(
                    father_img, child_img,
                    model_name=config.MODEL_NAME,
                    detector_backend=config.DETECTOR_BACKEND,
                    enforce_detection=False,  # Don't enforce face detection
                    distance_metric="euclidean"
                )
                
                # Compare mother and child
                maternal_result = DeepFace.verify(
                    mother_img, child_img,
                    model_name=config.MODEL_NAME,
                    detector_backend=config.DETECTOR_BACKEND,
                    enforce_detection=False,  # Don't enforce face detection
                    distance_metric="euclidean"
                )
                
                # Determine resemblance
                if paternal_result["distance"] < maternal_result["distance"]:
                    resemblance = "father"
                else:
                    resemblance = "mother"
                    
                # Construct result
                result = {
                    "paternal_result": paternal_result,
                    "maternal_result": maternal_result,
                    "resemblance": resemblance,
                    "face_detection_enforced": False
                }
                
                return JSONResponse(result)
            except Exception as e:
                raise HTTPException(500, f"Error during face comparison with enforce_detection=False: {str(e)}")
        raise HTTPException(500, f"Error during face comparison: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Unexpected error: {str(e)}")

async def identify_faces(target: UploadFile):
    """Identify faces in an image by comparing against the reference database."""
    # Read and process the uploaded image
    image_bytes = await target.read()
    image = bgr_from_upload(image_bytes)
    
    # Extract faces from the image
    faces = DeepFace.extract_faces(image, detector_backend=config.DETECTOR_BACKEND)

    if not faces:
        raise HTTPException(422, "No faces were detected in the image")

    # Prepare reference info
    ref_path = str(config.REFERENCE_DIR)
    
    # List reference images
    ref_contents = []
    ref_labels = []
    if os.path.exists(ref_path):
        for root, dirs, files in os.walk(ref_path):
            for file in files:
                # Skip non-image files
                if any(file.lower().endswith(ext) for ext in config.ALLOWED_EXT):
                    ref_contents.append(os.path.join(root, file))
                    # Extract label from path
                    label = os.path.basename(root)
                    if label not in ref_labels and not label.startswith('ds_model'):
                        ref_labels.append(label)
    
    # Count reference photos per label
    label_counts = {}
    for label in ref_labels:
        label_path = os.path.join(ref_path, label)
        if os.path.isdir(label_path):
            image_files = glob.glob(os.path.join(label_path, "*.jpg")) + \
                         glob.glob(os.path.join(label_path, "*.jpeg")) + \
                         glob.glob(os.path.join(label_path, "*.png"))
            label_counts[label] = len(image_files)
    
    # Include in response for debugging
    debug_info = {
        "reference_path": ref_path,
        "reference_exists": os.path.exists(ref_path),
        "reference_contents": ref_contents[:20],  # Limit to 20 files
        "reference_count": len(ref_contents),
        "reference_labels": ref_labels,
        "label_counts": label_counts,
        "repr_file_exists": os.path.exists(os.path.join(ref_path, "representations_arcface.pkl"))
    }

    # First make sure we have a valid reference database
    if not ref_contents:
        return [{"bbox": face["facial_area"], "label": "unknown", "distance": None, 
                 "error": "No reference photos in database", "debug": debug_info} 
                for face in faces]
    
    # Ensure we have a fresh representation database
    rebuild_reference_db()
    
    matches = []
    for i, face in enumerate(faces):
        try:
            # Save the face temporarily to use with DeepFace.find
            face_img = face["face"]  # This is a numpy array
            
            # Fix image format - convert from float to uint8 if needed
            if face_img.dtype == np.float64 or face_img.dtype == np.float32:
                face_img = (face_img * 255).astype(np.uint8)
            
            # Ensure correct format for JPEG
            if len(face_img.shape) != 3 or face_img.shape[2] != 3:
                if len(face_img.shape) == 2:  # Grayscale
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
                elif face_img.shape[2] == 4:  # RGBA
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_RGBA2BGR)
            
            temp_file = f"/tmp/temp_face_{i}.jpg"
            success = cv2.imwrite(temp_file, face_img)
            
            if not success:
                raise ValueError(f"Failed to save temporary face image to {temp_file}")
            
            # Explicitly build the model first
            _ = DeepFace.build_model(config.MODEL_NAME)
            
            # Use DeepFace.find against the reference folder using the temp file
            raw = DeepFace.find(
                img_path=temp_file,
                db_path=str(config.REFERENCE_DIR),
                model_name=config.MODEL_NAME,
                detector_backend=config.DETECTOR_BACKEND,
                enforce_detection=False,
                distance_metric="cosine",
                threshold=config.IDENTITY_THRESHOLD,
                silent=True
            )
            
            # Clean up temp file
            os.remove(temp_file)
            
            # Process results
            if isinstance(raw, list) and len(raw) > 0:
                res = raw[0]
            else:
                res = raw
                
            label, dist = "unknown", None
            # If we got results and the closest match is below threshold, extract label
            if not res.empty and res.iloc[0].distance <= config.IDENTITY_THRESHOLD:
                # Extract label from identity path
                identity_path = res.iloc[0].identity
                label = os.path.basename(os.path.dirname(identity_path))
                dist = float(res.iloc[0].distance)

        except ValueError as e:
            label, dist = "unknown", None
        except Exception as e:
            label, dist = "unknown", None

        matches.append({
            "bbox": face["facial_area"],
            "label": label,
            "distance": dist
        })
    
    # Add debug info to the first match (if any)
    if matches:
        matches[0]["debug"] = debug_info
    
    return matches

async def analyze_face(photo: UploadFile):
    """Extract facial attributes (age, gender, emotion, race)."""
    img = bgr_from_upload(await photo.read())
    result = DeepFace.analyze(
        img,
        actions=["age", "gender", "emotion", "race"],
        detector_backend=config.DETECTOR_BACKEND
    )
    # Ensure all outputs are JSON serializable (convert numpy types to Python types)
    return to_serializable(result)
