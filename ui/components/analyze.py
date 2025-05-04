"""Face attribute analysis component for DeepFace Suite."""

import streamlit as st
from typing import Dict, List, Optional
from PIL import Image

from utils.api import analyze_faces
from utils.image import create_annotated_image, display_interactive_image
from utils.ui import display_json, format_face_attributes, create_face_table, create_attribute_details, SUPPORTED_FORMATS

def analyze_attributes_ui(api_url: str) -> None:
    """UI for analyzing facial attributes like age, gender, emotion, and race.
    
    Args:
        api_url: Base URL for the API
    """
    st.subheader("Analyze Face Attributes")
    
    # Form instructions
    st.write("""
    Upload a photo to analyze facial attributes like age, gender, emotion, and race.
    Works with both individual portraits and group photos.
    """)
    
    # Form inputs
    photo = st.file_uploader("Portrait or group photo", type=SUPPORTED_FORMATS, key="analyze_photo_uploader")
    
    # Form validation
    if not photo or not st.button("Analyze", key="analyze_button"):
        return
    
    # Process analysis
    with st.spinner("Analyzing faces..."):
        data = analyze_faces(photo, api_url)
        
        if not data:
            st.error("No faces were detected or there was an API error.")
            return
            
        # Prepare face data
        st.success(f"Found and analyzed {len(data)} faces in the image")
        
        # Display annotated image
        img = Image.open(photo).convert("RGB")
        
        # Create face IDs with dominant attributes
        face_ids = {}
        for i, face in enumerate(data):
            gender = face.get("dominant_gender", "")
            age = face.get("age", "")
            emotion = face.get("dominant_emotion", "")
            face_ids[i] = f"{gender} {age}y {emotion}"
            
        # Create annotated image
        annotated_img, face_info = create_annotated_image(img, data, face_ids)
        
        # Display the interactive image with hover
        st.subheader("Analysis Results")
        st.info("Hover over faces to see attribute details")
        
        # Create tooltip formatter for attribute display with detailed percentages
        def attr_tooltip(face: Dict) -> str:
            gender = face.get("dominant_gender", "")
            age = face.get("age", "")
            emotion = face.get("dominant_emotion", "")
            race = face.get("dominant_race", "")
            
            lines = []
            # Add header with main attributes
            lines.append(f"<b>{gender}, {age} years, {emotion}</b>")
            
            # Gender details
            if "gender" in face and isinstance(face["gender"], dict):
                lines.append("<b>Gender:</b>")
                for g, value in face["gender"].items():
                    # Format to 2 decimal places
                    lines.append(f"- {g}: {value:.2f}%")
            
            # Emotion details 
            if "emotion" in face and isinstance(face["emotion"], dict):
                lines.append("<b>Emotion:</b>")
                # Sort emotions by confidence descending, and show top 3
                emotions = sorted(face["emotion"].items(), key=lambda x: x[1], reverse=True)[:3]
                for emotion, value in emotions:
                    if value > 0.01:  # Only show emotions with some confidence
                        lines.append(f"- {emotion}: {value:.2f}%")
            
            # Race details
            if "race" in face and isinstance(face["race"], dict):
                lines.append("<b>Race/Ethnicity:</b>")
                # Sort races by confidence descending, and show top 3
                races = sorted(face["race"].items(), key=lambda x: x[1], reverse=True)[:3]
                for race, value in races:
                    if value > 0.5:  # Only show races with some confidence
                        lines.append(f"- {race}: {value:.2f}%")
            
            # Add face confidence
            if "face_confidence" in face:
                confidence = face["face_confidence"] * 100
                lines.append(f"<b>Detection confidence:</b> {confidence:.2f}%")
                
            return "<br>".join(lines)
        
        # Display interactive image
        display_interactive_image(img, data, attr_tooltip)
        
        # Also show static image as fallback
        with st.expander("Static Image (with labels)"):
            st.image(annotated_img, caption="Face Analysis Results", use_column_width=True)
        
        # Create summary table
        st.subheader("Analysis Summary")
        create_face_table(data)
        
        # Display details for each face
        st.subheader("Detailed Analysis")
        st.info("👇 Click on the expanders below to see detailed analysis for each face")
        
        for i, face in enumerate(data):
            with st.expander(f"Face #{i+1} Details"):
                create_attribute_details(face)
        
        # Display full raw data
        with st.expander("View detailed analysis data"):
            display_json(data)
