"""Face comparison component for DeepFace Suite."""

import streamlit as st
from typing import Dict, Optional
from PIL import Image

from utils.api import compare_faces
from utils.image import create_annotated_image, display_interactive_image
from utils.ui import display_json, format_face_attributes, SUPPORTED_FORMATS

def compare_photos_ui(api_url: str) -> None:
    """UI for comparing two face photos.
    
    Args:
        api_url: Base URL for the API
    """
    st.subheader("Compare Two Photos")
    
    # Form instructions
    st.write("""
    Upload two photos to compare if they are the same person.
    """)
    
    # Form inputs in two columns
    col1, col2 = st.columns(2)
    with col1:
        img1 = st.file_uploader("Photo A", type=SUPPORTED_FORMATS, key="img1")
    with col2:
        img2 = st.file_uploader("Photo B", type=SUPPORTED_FORMATS, key="img2")
    
    # Form validation
    if not img1 or not img2 or not st.button("Compare"):
        return
    
    # Process comparison
    with st.spinner("Comparing faces..."):
        result = compare_faces(img1, img2, api_url)
        
        if not result:
            st.error("Could not compare faces. Check if both images contain detectable faces.")
            return
            
        # Display match status prominently
        if result.get("verified", False):
            st.success("✓ Match verified! These appear to be the same person.")
        else:
            st.warning("✗ No match. These appear to be different people.")
        
        # Load images
        image1 = Image.open(img1).convert("RGB")
        image2 = Image.open(img2).convert("RGB")
        
        # Check if we have face detection data in the response
        face1_data = []
        face2_data = []
        
        # Extract face regions from the correct location in the JSON
        # The response contains facial_areas with img1 and img2 keys
        if "facial_areas" in result:
            if "img1" in result["facial_areas"]:
                face1_data = [{
                    "region": result["facial_areas"]["img1"],
                    "verified": result.get("verified", False),
                    "distance": result.get("distance", 0),
                }]
                
            if "img2" in result["facial_areas"]:
                face2_data = [{
                    "region": result["facial_areas"]["img2"],
                    "verified": result.get("verified", False),
                    "distance": result.get("distance", 0),
                }]
        
        # Display the results
        st.subheader("Comparison Results")
        
        if face1_data and face2_data:
            # Show interactive images with hover tooltips
            st.info("Hover over faces to see match information")
            
            # Create tooltip formatter
            def tooltip_formatter(face: Dict) -> str:
                status = "Match ✓" if face.get("verified", False) else "No match ✗"
                confidence = (1 - face.get("distance", 0)) * 100
                return f"<b>{status}</b><br>Confidence: {confidence:.1f}%"
            
            # Display images with tooltips
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Photo A**")
                display_interactive_image(image1, face1_data, tooltip_formatter, width=400)
                
            with col2:
                st.write("**Photo B**")
                display_interactive_image(image2, face2_data, tooltip_formatter, width=400)
            
            # Also create and display static annotated images
            annotated_img1, _ = create_annotated_image(image1, face1_data)
            annotated_img2, _ = create_annotated_image(image2, face2_data)
            
            # Show static images
            with st.expander("Static Images (with labels)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.image(annotated_img1, caption="Photo A", use_column_width=True)
                with col2:
                    st.image(annotated_img2, caption="Photo B", use_column_width=True)
        else:
            # Just show the original images
            col1, col2 = st.columns(2)
            with col1:
                st.image(image1, caption="Photo A", use_column_width=True)
            with col2:
                st.image(image2, caption="Photo B", use_column_width=True)
            
            st.warning("No face regions returned from API. Showing original images.")
            
        # Display match metrics
        if "distance" in result:
            st.subheader("Match Metrics")
            confidence = (1 - result["distance"]) * 100
            st.metric("Match Confidence", f"{confidence:.1f}%")
            st.progress(confidence / 100)
            
            # Add threshold information if available
            if "threshold" in result:
                threshold = result.get("threshold", 0.6)
                st.write(f"Match threshold: {threshold} (lower distance = better match)")
                
        # Display raw data
        with st.expander("View comparison details"):
            st.markdown(f"**Model Used:** {result.get('model', 'Unknown')}")
            display_json(result)
