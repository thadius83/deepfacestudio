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

async def identify_faces(target: UploadFile, threshold: float = None):
    """Identify faces in an image by comparing against the reference database.
    
    Args:
        target: The uploaded image file
        threshold: Optional custom distance threshold (0-1, lower = more confident match)
                  Overrides the default threshold in config if provided
    """
    # Read and process the uploaded image
    image_bytes = await target.read()
    image = bgr_from_upload(image_bytes)
    
    # Call the core detection and identification function
    return detect_and_identify(image, threshold)

def detect_and_identify(image, threshold: float = None):
    """Core detection and identification function that operates on raw image data.
    
    Args:
        image: BGR image as numpy array
        threshold: Optional distance threshold
        
    Returns:
        List of face matches
    """
    # Use provided threshold or default from config
    identity_threshold = threshold if threshold is not None else config.IDENTITY_THRESHOLD
    print(f"Using identity threshold: {identity_threshold}")
    
    # Extract faces from the image
    faces = DeepFace.extract_faces(image, detector_backend=config.DETECTOR_BACKEND)

    if not faces:
        return [{"error": "No faces were detected in the image"}]

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
    
    # Only rebuild database if needed (if pickle file doesn't exist)
    file_parts = [
        "ds", 
        "model", 
        config.MODEL_NAME.lower(), 
        "detector", 
        config.DETECTOR_BACKEND,
        "aligned",
        "normalization", 
        "base",
        "expand", 
        "0"
    ]
    file_name = "_".join(file_parts) + ".pkl"
    file_name = file_name.replace("-", "").lower()
    
    pkl_path = os.path.join(str(config.REFERENCE_DIR), file_name)
    
    # Only rebuild if the pickle file doesn't exist
    if not os.path.exists(pkl_path):
        print(f"Rebuilding reference database (pickle file not found: {pkl_path})")
        rebuild_reference_db()
    
    matches = []
    for i, face in enumerate(faces):
        try:
            # Save the face temporarily to use with DeepFace.find
            print(f"Processing face #{i+1}")
            face_img = face["face"]  # This is a numpy array
            
            # Fix image format - convert from float to uint8 if needed
            if face_img.dtype == np.float64 or face_img.dtype == np.float32:
                face_img = (face_img * 255).astype(np.uint8)
                print(f"Converted face image from float to uint8")
            
            # Ensure correct format for JPEG
            if len(face_img.shape) != 3 or face_img.shape[2] != 3:
                print(f"Converting face image format: shape={face_img.shape}")
                if len(face_img.shape) == 2:  # Grayscale
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
                    print("Converted grayscale to BGR")
                elif face_img.shape[2] == 4:  # RGBA
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_RGBA2BGR)
                    print("Converted RGBA to BGR")
            
            # Use a temporary directory that's more likely to exist across platforms
            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f"temp_face_{i}.jpg")
            print(f"Saving temp face to: {temp_file}")

            success = cv2.imwrite(temp_file, face_img)
            if not success:
                raise ValueError(f"Failed to save temporary face image to {temp_file}")

            # Verify the file exists before proceeding
            if not os.path.exists(temp_file):
                raise ValueError(f"Temp file was not created at {temp_file}")
                
            print(f"Temp file created successfully: {os.path.getsize(temp_file)} bytes")

            # Explicitly build the model first
            print(f"Building model: {config.MODEL_NAME}")
            _ = DeepFace.build_model(config.MODEL_NAME)

            print("Preparing to identify face...")
            print(f"Using threshold: {identity_threshold}")

            # Try the DeepFace.find operation directly
            try:
                print("Attempting DeepFace.find operation")
                dfs = DeepFace.find(
                    img_path=temp_file,
                    db_path=ref_path,
                    model_name=config.MODEL_NAME,
                    detector_backend=config.DETECTOR_BACKEND,
                    distance_metric="cosine",
                    enforce_detection=False,
                    align=True
                )
                
                # Process the result
                if dfs and len(dfs) > 0 and not dfs[0].empty:
                    # Find the best match
                    closest_match = dfs[0].iloc[0]
                    print(f"Find operation found match: {closest_match['identity']}")
                    match_distance = closest_match["distance"]
                    
                    # Extract identity path
                    identity_path = closest_match["identity"]
                    # Get the label from the path
                    label_name = os.path.basename(os.path.dirname(identity_path))
                    
                    if match_distance <= identity_threshold:
                        label = label_name
                        dist = match_distance
                    else:
                        print(f"Match distance {match_distance} exceeds threshold {identity_threshold}")
                        label, dist = "unknown", None
                else:
                    print("DeepFace.find returned no matches")
                    label, dist = "unknown", None
                    
                # Clean up early
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        print(f"Removed temp file: {temp_file}")
                except Exception as e:
                    print(f"Error removing temp file: {str(e)}")
                
                # Skip the manual comparison if we already have a match
                if label != "unknown":
                    print(f"Using match from DeepFace.find: {label}")
                    matches.append({
                        "bbox": face["facial_area"],
                        "label": label,
                        "distance": dist
                    })
                    continue
            except Exception as e:
                print(f"DeepFace.find error: {str(e)}")
                print("Falling back to manual comparison method")

            print("Starting manual comparison method...")
            print(f"Using threshold: {identity_threshold}")
            
            # ALTERNATIVE APPROACH: Directly represent the face and compare with reference images
            print("Using direct embedding comparison approach instead of DeepFace.find")
            
            # Get the embedding for this face
            print("Generating embedding for detected face...")
            from deepface.commons import functions
            
            # Use DeepFace.represent to get embedding
            embedding_objs = DeepFace.represent(
                img_path=temp_file,
                model_name=config.MODEL_NAME,
                detector_backend=config.DETECTOR_BACKEND,
                enforce_detection=False,
                align=True
            )
            
            if not embedding_objs:
                print("No embeddings generated for face")
                label, dist = "unknown", None
            else:
                # Get the embedding vector 
                embedding = embedding_objs[0]["embedding"]
                print(f"Successfully generated embedding: length={len(embedding)}")
                
                # Clean up temp file early since we're done with it
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        print(f"Removed temp file: {temp_file}")
                except Exception as e:
                    print(f"Error removing temp file: {str(e)}")
                
                # List all reference photos by label
                print("Comparing with reference images directly...")
                best_match = {"label": "unknown", "distance": 1.0}
                
                # Manual matching against reference database
                for label_name in ref_labels:
                    label_dir = os.path.join(ref_path, label_name)
                    print(f"Checking label: {label_name}")
                    
                    # Skip if not a directory
                    if not os.path.isdir(label_dir):
                        continue
                    
                    # Process at most 5 images per label to avoid long processing
                    image_files = []
                    for ext in config.ALLOWED_EXT:
                        image_files.extend(glob.glob(os.path.join(label_dir, f"*{ext}")))
                    
                    # Limit number of files to process
                    image_files = image_files[:5]
                    
                    for ref_img_path in image_files:
                        print(f"  Comparing with: {os.path.basename(ref_img_path)}")
                        
                        try:
                            # Get embedding for reference image
                            ref_embedding_objs = DeepFace.represent(
                                img_path=ref_img_path,
                                model_name=config.MODEL_NAME,
                                detector_backend=config.DETECTOR_BACKEND,
                                enforce_detection=False,
                                align=True
                            )
                            
                            if ref_embedding_objs:
                                ref_embedding = ref_embedding_objs[0]["embedding"]
                                
                                # Calculate distance
                                from deepface.commons.distance import findCosineDistance
                                distance = findCosineDistance(embedding, ref_embedding)
                                
                                print(f"  Distance: {distance}")
                                
                                # Check if this is the best match so far
                                if distance < best_match["distance"]:
                                    best_match = {
                                        "label": label_name,
                                        "distance": distance,
                                        "file": os.path.basename(ref_img_path)
                                    }
                        except Exception as e:
                            print(f"  Error processing reference image: {str(e)}")
                
                # Check if the best match meets the threshold
                print(f"Best match: {best_match['label']} with distance {best_match['distance']}")
                if best_match["distance"] <= identity_threshold:
                    label = best_match["label"]
                    dist = best_match["distance"]
                else:
                    label, dist = "unknown", None

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
    print("analyze_face called")
    img_bytes = await photo.read()
    print(f"Received image: {len(img_bytes)} bytes")
    img = bgr_from_upload(img_bytes)
    print(f"Converted to BGR image: shape={img.shape}, dtype={img.dtype}")
    
    print("Starting DeepFace.analyze...")
    try:
        result = DeepFace.analyze(
            img,
            actions=["age", "gender", "emotion", "race"],
            detector_backend=config.DETECTOR_BACKEND
        )
        print("DeepFace.analyze completed successfully")
        # Ensure all outputs are JSON serializable (convert numpy types to Python types)
        return to_serializable(result)
    except Exception as e:
        print(f"DeepFace.analyze error: {str(e)}")
        raise
