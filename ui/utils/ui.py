"""UI utilities for DeepFace Suite."""

import json
import streamlit as st
from typing import Dict, List, Any, Optional

# Constants
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp"]

def display_json(data: Any) -> None:
    """Display formatted JSON data.
    
    Args:
        data: Data to display as JSON
    """
    if data:
        st.code(json.dumps(data, indent=2), language="json")

def format_face_attributes(face: Dict) -> str:
    """Format face attributes for display.
    
    Args:
        face: Dictionary containing face attributes
        
    Returns:
        Formatted HTML string for display
    """
    lines = []
    
    # Age
    if "age" in face:
        lines.append(f"**Age:** {face['age']} years")
    
    # Gender
    if "dominant_gender" in face:
        lines.append(f"**Gender:** {face['dominant_gender']}")
    
    # Emotion
    if "dominant_emotion" in face:
        lines.append(f"**Emotion:** {face['dominant_emotion']}")
    
    # Race
    if "dominant_race" in face:
        lines.append(f"**Race:** {face['dominant_race']}")
    
    # Person identification
    if "label" in face:
        lines.append(f"**Person:** {face['label']}")
    
    # Match info
    if "distance" in face and face["distance"] is not None:
        lines.append(f"**Match confidence:** {(1-face['distance'])*100:.1f}%")
    
    # Verification 
    if "verified" in face:
        status = "✅ Verified" if face["verified"] else "❌ Not verified"
        lines.append(f"**Status:** {status}")
    
    # Processing status
    if "processing_status" in face:
        lines.append(f"**Processing:** {face['processing_status']}")
    
    return "\n".join(lines)

def format_face_tooltip(face: Dict) -> str:
    """Format face attributes for tooltip display.
    
    Args:
        face: Dictionary containing face attributes
        
    Returns:
        Formatted HTML string for tooltips
    """
    lines = []
    
    # Person identification (if available)
    if "label" in face and face["label"] != "unknown":
        lines.append(f"<b>{face['label']}</b>")
    
    # Age
    if "age" in face:
        lines.append(f"Age: {face['age']} yrs")
    
    # Gender
    if "dominant_gender" in face:
        lines.append(f"Gender: {face['dominant_gender']}")
    
    # Emotion
    if "dominant_emotion" in face:
        lines.append(f"Emotion: {face['dominant_emotion']}")
    
    # Match confidence (in percentage)
    if "distance" in face and face["distance"] is not None:
        confidence = (1 - face["distance"]) * 100
        lines.append(f"Confidence: {confidence:.1f}%")
    
    # If no specific attributes, add a generic label
    if not lines:
        lines.append("Face detected")
    
    return "<br>".join(lines)

def create_face_table(faces_data: List[Dict]) -> None:
    """Create a summary table of face data.
    
    Args:
        faces_data: List of face data dictionaries
    """
    if not faces_data:
        return
        
    # Create summary table data
    summary_data = []
    for i, face in enumerate(faces_data):
        row = {"Face": i+1}
        
        # Add common attributes
        if "age" in face:
            row["Age"] = face.get("age", "N/A")
        if "dominant_gender" in face:
            row["Gender"] = face.get("dominant_gender", "N/A")
        if "dominant_emotion" in face:
            row["Emotion"] = face.get("dominant_emotion", "N/A")
        if "dominant_race" in face:
            row["Race"] = face.get("dominant_race", "N/A")
        if "label" in face:
            row["Person"] = face.get("label", "Unknown")
        if "verified" in face:
            row["Match"] = "✓" if face.get("verified", False) else "✗"
        if "distance" in face and face["distance"] is not None:
            row["Confidence"] = f"{(1-face['distance'])*100:.1f}%"
            
        summary_data.append(row)
    
    # Display as dataframe
    st.dataframe(summary_data)

def create_attribute_details(face: Dict) -> None:
    """Create detailed attribute view with visualizations.
    
    Args:
        face: Face data dictionary
    """
    cols = st.columns(2)
    
    # Left column - main attributes
    with cols[0]:
        st.markdown("#### Main Attributes")
        st.markdown(f"**Age:** {face.get('age', 'N/A')} years")
        st.markdown(f"**Gender:** {face.get('dominant_gender', 'N/A')}")
        st.markdown(f"**Emotion:** {face.get('dominant_emotion', 'N/A')}")
        st.markdown(f"**Race:** {face.get('dominant_race', 'N/A')}")
    
    # Right column - detailed percentages
    with cols[1]:
        # Gender distribution
        if "gender" in face and isinstance(face["gender"], dict):
            st.markdown("#### Gender Distribution")
            for gender, confidence in face["gender"].items():
                st.progress(min(confidence / 100, 1.0))
                st.markdown(f"{gender}: {confidence:.1f}%")
        
        # Emotion distribution
        if "emotion" in face and isinstance(face["emotion"], dict):
            st.markdown("#### Emotion Distribution")
            # Filter out minor emotions for cleaner display
            emotions = {k: v for k, v in face["emotion"].items() if v > 1.0}
            for emotion, confidence in emotions.items():
                st.progress(min(confidence / 100, 1.0))
                st.markdown(f"{emotion}: {confidence:.1f}%")
