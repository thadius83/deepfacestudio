"""Reference photo management component for DeepFace Suite."""

import streamlit as st
from typing import List, Dict, Optional
from PIL import Image

from utils.api import add_reference_photos
from utils.ui import display_json, SUPPORTED_FORMATS

def reference_upload_ui(api_url: str) -> None:
    """UI for adding reference photos of individuals.
    
    Args:
        api_url: Base URL for the API
    """
    st.subheader("Add Reference Photos")
    
    # Form instructions
    st.write("""
    Upload reference photos of people you want to identify later. 
    Add a unique label for each person (no spaces).
    """)
    
    # Form inputs
    label = st.text_input("Person label (unique, no spaces)")
    files = st.file_uploader(
        "Reference image(s)", 
        type=SUPPORTED_FORMATS,
        accept_multiple_files=True
    )
    
    # Form validation
    if not st.button("Upload") or not label or not files:
        return
    
    # Process upload
    with st.spinner("Uploading…"):
        result = add_reference_photos(label, files, api_url)
        
        if result:
            st.success(f"Successfully uploaded {len(files)} reference photos for '{label}'")
            
            # Preview the uploaded images
            cols = st.columns(min(len(files), 4))
            for i, file in enumerate(files):
                with cols[i % len(cols)]:
                    st.image(file, caption=f"{label} - Image {i+1}", use_column_width=True)
            
            with st.expander("API Response Details"):
                display_json(result)
