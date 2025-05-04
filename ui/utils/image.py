"""Image processing utilities for DeepFace Suite."""

import io
import base64
from typing import Dict, List, Tuple, Optional, Callable, Any
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import streamlit as st

# Constants
LABEL_BACKGROUND_COLORS = {
    "match": (0, 255, 0),    # Green for matches
    "detect": (255, 255, 0),  # Yellow for detections
    "error": (255, 0, 0)      # Red for errors
}

def pil_to_bytes(image: Image.Image, format: str = "JPEG") -> bytes:
    """Convert PIL Image to bytes.
    
    Args:
        image: PIL Image object
        format: Image format (default: JPEG)
        
    Returns:
        Image bytes
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()

def crop_face(image: Image.Image, bbox: Dict[str, int]) -> Image.Image:
    """Crop a face from an image based on bounding box.
    
    Args:
        image: PIL Image object
        bbox: Dict with x, y, w, h coordinates
        
    Returns:
        Cropped PIL Image
    """
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    return image.crop((x, y, x + w, y + h))

def draw_bbox_with_label(
    image: Image.Image, 
    bbox: Dict[str, int], 
    label: str, 
    status: str = "detect"
) -> Image.Image:
    """Draw bounding box with label on image.
    
    Args:
        image: PIL Image object
        bbox: Dict with x, y, w, h coordinates
        label: Text to display above box
        status: Box status (match, detect, error) for color
        
    Returns:
        PIL Image with box and label
    """
    # Create a copy to avoid modifying the original
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    
    # Extract coordinates
    x, y = bbox["x"], bbox["y"]
    w, h = bbox["w"], bbox["h"]
    
    # Get color based on status
    color = LABEL_BACKGROUND_COLORS.get(status, LABEL_BACKGROUND_COLORS["detect"])
    
    # Draw rectangle
    draw.rectangle(
        [(x, y), (x + w, y + h)],
        outline=color,
        width=3
    )
    
    # Draw label background
    text_w, text_h = draw.textbbox((0, 0), label)[2:4]
    draw.rectangle(
        [(x, y - text_h - 4), (x + text_w + 4, y)],
        fill=color
    )
    
    # Draw text
    text_color = (0, 0, 0)  # Black text on colored background
    draw.text((x + 2, y - text_h - 2), label, fill=text_color)
    
    return image_copy

def create_annotated_image(
    image: Image.Image,
    faces_data: List[Dict],
    face_ids: Optional[Dict[int, str]] = None,
) -> Tuple[Image.Image, List[Dict]]:
    """Create an image with face annotations.
    
    Args:
        image: PIL Image object
        faces_data: List of face data dictionaries
        face_ids: Optional mapping of face IDs
        
    Returns:
        Tuple of (annotated image, face info list)
    """
    # Make a copy of the image
    annotated_img = image.copy()
    
    # List to store face info for display
    face_info = []
    
    # Draw bounding boxes
    for i, face in enumerate(faces_data):
        # Skip if no bounding box info
        if not ("bbox" in face or "region" in face):
            continue
            
        # Extract bounding box
        bbox = face.get("bbox") or face.get("region")
            
        # Generate face ID
        face_id = f"Face #{i+1}" if face_ids is None else face_ids.get(i, f"Face #{i+1}")
        
        # Determine status based on verification or identification
        status = "match" if (face.get("verified", False) or 
                            (face.get("label", "unknown") != "unknown" and 
                             face.get("distance") is not None)) else "detect"
        
        # Add check mark for verified faces
        if face.get("verified", False):
            face_id = "✓ " + face_id
        
        # Draw bbox with face ID
        annotated_img = draw_bbox_with_label(
            annotated_img, 
            bbox, 
            face_id,
            status
        )
        
        # Save face info for display
        face_info.append({
            "id": face_id,
            "data": face
        })
    
    return annotated_img, face_info

def create_data_uri(image: Image.Image, format: str = "JPEG") -> str:
    """Create a data URI for an image.
    
    Args:
        image: PIL Image object
        format: Image format (default: JPEG)
        
    Returns:
        Data URI string
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"

def create_interactive_html(
    image: Image.Image,
    faces_data: List[Dict],
    tooltip_fn: Callable[[Dict], str],
    width: int = 800
) -> str:
    """Create HTML with interactive hover areas for faces.
    
    Args:
        image: PIL Image object
        faces_data: List of face data dictionaries
        tooltip_fn: Function to generate tooltip content from face data
        width: Image width in pixels
        
    Returns:
        HTML string with interactive elements
    """
    # Resize image maintaining aspect ratio
    img_width, img_height = image.size
    scale_factor = width / img_width
    height = int(img_height * scale_factor)
    resized_img = image.resize((width, height), Image.LANCZOS)
    
    # Create data URI for the image
    img_uri = create_data_uri(resized_img)
    
    # Start HTML
    html = f"""
    <div style="position: relative; width: {width}px; height: {height}px;">
      <img src="{img_uri}" style="width: 100%; height: 100%;"/>
    """
    
    # Add hover areas for each face
    for i, face in enumerate(faces_data):
        # Skip if no bounding box info
        if not ("bbox" in face or "region" in face):
            continue
            
        # Extract bounding box
        bbox = face.get("bbox") or face.get("region")
        
        # Scale bounding box to match resized image
        x = int(bbox["x"] * scale_factor)
        y = int(bbox["y"] * scale_factor)
        w = int(bbox["w"] * scale_factor)
        h = int(bbox["h"] * scale_factor)
        
        # Generate tooltip content
        tooltip = tooltip_fn(face).replace("\n", "<br/>")
        
        # Add hover area with tooltip
        html += f"""
        <div class="face-box" 
             data-face-id="{i}" 
             style="position: absolute; left: {x}px; top: {y}px; width: {w}px; height: {h}px; 
                    border: 3px solid {'green' if (face.get('verified', False) or (face.get('label', 'unknown') != 'unknown' and face.get('distance') is not None)) else 'yellow'}; 
                    cursor: pointer;"
             onmouseenter="showTooltip(event, {i})"
             onmouseleave="hideTooltip({i})">
        </div>
        <div class="tooltip" 
             id="tooltip-{i}" 
             style="display: none; position: absolute; background: rgba(0,0,0,0.8); 
                    color: white; padding: 8px; border-radius: 4px; 
                    max-width: 300px; z-index: 100; pointer-events: none;">
          {tooltip}
        </div>
        """
    
    # Add JavaScript for tooltip handling
    html += """
    <script>
    function showTooltip(event, id) {
        const tooltip = document.getElementById('tooltip-' + id);
        tooltip.style.display = 'block';
        tooltip.style.left = (event.clientX + 10) + 'px';
        tooltip.style.top = (event.clientY + 10) + 'px';
    }
    
    function hideTooltip(id) {
        document.getElementById('tooltip-' + id).style.display = 'none';
    }
    
    // Add event listener for mousemove to update tooltip position
    document.querySelectorAll('.face-box').forEach(box => {
        box.addEventListener('mousemove', function(event) {
            const id = this.getAttribute('data-face-id');
            const tooltip = document.getElementById('tooltip-' + id);
            tooltip.style.left = (event.clientX + 10) + 'px';
            tooltip.style.top = (event.clientY + 10) + 'px';
        });
    });
    </script>
    """
    
    # Close HTML container
    html += "</div>"
    
    return html

def display_interactive_image(
    image: Image.Image,
    faces_data: List[Dict],
    tooltip_fn: Callable[[Dict], str],
    width: int = 800
) -> None:
    """Display an interactive image with face hover tooltips.
    
    Args:
        image: PIL Image object
        faces_data: List of face data dictionaries
        tooltip_fn: Function to generate tooltip content from face data
        width: Image width in pixels
    """
    html = create_interactive_html(image, faces_data, tooltip_fn, width)
    st.components.v1.html(html, height=(image.height * width // image.width) + 50)
