"""API utilities for DeepFace Suite."""

import requests
import streamlit as st
from typing import Dict, List, Union, Optional, Any, Tuple

def api_request(
    endpoint: str, 
    files: Union[Dict[str, Any], List[Tuple]], 
    api_url: str,
    method: str = "post",
    params: Dict[str, str] = None
) -> Optional[Dict]:
    """Make API request to the backend service with enhanced error handling.
    
    Args:
        endpoint: API endpoint path
        files: Dictionary or list of files to upload
        api_url: Base URL for the API
        method: HTTP method (default: post)
        params: Optional query parameters
        
    Returns:
        JSON response or None on error
    """
    try:
        url = f"{api_url}/{endpoint}"
        
        # Log the request (in development mode)
        with st.expander("API Request Details", expanded=False):
            st.write(f"Endpoint: {url}")
            if isinstance(files, dict):
                st.write(f"Files: {list(files.keys())}")
            else:
                st.write(f"Files: {len(files)} file(s)")
            if params:
                st.write(f"Params: {params}")
        
        response = requests.post(url, files=files, params=params)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            st.error(f"Response Status: {e.response.status_code}")
            try:
                st.error(f"Response Text: {e.response.text}")
            except:
                pass
        return None

def add_reference_photos(label: str, files: List, api_url: str) -> Optional[Dict]:
    """Add reference photos for a person.
    
    Args:
        label: Person label (unique identifier)
        files: List of file objects
        api_url: Base URL for the API
        
    Returns:
        API response or None on error
    """
    try:
        url = f"{api_url}/reference/{label}"
        # Log request details for debugging
        st.info(f"Uploading {len(files)} files to {url}")
        
        # Display debugging info about the files being uploaded
        for i, f in enumerate(files):
            with st.expander(f"File {i+1} details", expanded=False):
                st.write(f"Filename: {f.name}")
                st.write(f"Type: {f.type}")
                st.write(f"Size: {len(f.getvalue())} bytes")
                # Check for common image issues
                if len(f.getvalue()) < 1000:
                    st.warning(f"⚠️ Very small file: {len(f.getvalue())} bytes")
                
                # Preview image
                try:
                    st.image(f, caption=f"Preview: {f.name}", width=150)
                except Exception as e:
                    st.error(f"Cannot preview image: {str(e)}")
        
        # Make the request directly to handle multiple files with the same param name
        st.info("Sending files to API...")
        files_data = [('files', (f.name, f.getvalue(), f.type)) for f in files]
        response = requests.post(url, files=files_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            st.error(f"Response Status: {e.response.status_code}")
            try:
                st.error(f"Response Text: {e.response.text}")
            except:
                pass
        return None
    
def identify_faces(group_photo, api_url: str, threshold: float = None) -> Optional[List[Dict]]:
    """Identify all faces in a group photo against the reference database.
    
    Args:
        group_photo: File object with the group photo
        api_url: Base URL for the API
        threshold: Optional distance threshold (0-1, lower = more confident match)
        
    Returns:
        List of face matches or None on error
    """
    # Use the same simple pattern as the working analyze_faces function
    params = {}
    if threshold is not None:
        # Round to 2 decimal places to avoid weird floating point behavior
        threshold = round(threshold, 2)
        params = {"threshold": str(threshold)}
    
    # Just pass the file directly without any processing - like analyze_faces does
    return api_request("identify", {"target": group_photo}, api_url, params=params)

def compare_faces(img1, img2, api_url: str) -> Optional[Dict]:
    """Compare two face photos.
    
    Args:
        img1: First image file
        img2: Second image file
        api_url: Base URL for the API
        
    Returns:
        Comparison result or None on error
    """
    return api_request("compare", {"img1": img1, "img2": img2}, api_url)

def compare_face_with_reference(face_img, reference_img, api_url: str) -> Optional[Dict]:
    """Compare a face image with a reference image.
    
    Args:
        face_img: Face image data (can be bytes with filename and type)
        reference_img: Reference image file
        api_url: Base URL for the API
        
    Returns:
        Comparison result or None on error
    """
    # Create files parameter as a dictionary for more consistent behavior
    files = {}
    
    # Handle the face image (img1)
    if isinstance(face_img, tuple) and len(face_img) == 3:
        # Tuple of (filename, bytes, content_type)
        files["img1"] = face_img
    elif hasattr(face_img, 'name') and hasattr(face_img, 'getvalue') and hasattr(face_img, 'type'):
        # File-like object
        files["img1"] = (face_img.name, face_img.getvalue(), face_img.type)
    elif isinstance(face_img, bytes):
        # Raw bytes
        files["img1"] = ("face.jpg", face_img, "image/jpeg")
    else:
        raise ValueError("face_img must be a tuple, file-like object, or bytes")
    
    # Handle the reference image (img2)
    if hasattr(reference_img, 'name') and hasattr(reference_img, 'getvalue') and hasattr(reference_img, 'type'):
        # File-like object
        files["img2"] = (reference_img.name, reference_img.getvalue(), reference_img.type)
    elif isinstance(reference_img, bytes):
        # Raw bytes
        files["img2"] = ("reference.jpg", reference_img, "image/jpeg")
    elif isinstance(reference_img, tuple) and len(reference_img) == 3:
        # Tuple of (filename, bytes, content_type)
        files["img2"] = reference_img
    else:
        raise ValueError("reference_img must be a file-like object, bytes, or tuple")
    
    # Make the API request
    return api_request("compare", files, api_url)

def analyze_faces(photo, api_url: str) -> Optional[List[Dict]]:
    """Analyze facial attributes in a photo.
    
    Args:
        photo: File object with the photo
        api_url: Base URL for the API
        
    Returns:
        Analysis results or None on error
    """
    return api_request("analyze", {"photo": photo}, api_url)

def family_resemblance(father, child, mother, api_url: str) -> Optional[Dict]:
    """Compare child's face with both parents to determine resemblance.
    
    Args:
        father: Father's photo file
        child: Child's photo file
        mother: Mother's photo file
        api_url: Base URL for the API
        
    Returns:
        Resemblance results or None on error
    """
    return api_request("family-resemblance", {
        "father": father, 
        "child": child, 
        "mother": mother
    }, api_url)
