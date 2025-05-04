"""Face identification components for DeepFace Suite."""

import io
import streamlit as st
from typing import List, Dict, Optional, Any, Tuple
from PIL import Image

from utils.api import identify_faces, compare_face_with_reference
from utils.image import crop_face, create_annotated_image, display_interactive_image
from utils.ui import display_json, format_face_attributes, format_face_tooltip, create_face_table, SUPPORTED_FORMATS

def identify_group_ui(api_url: str) -> None:
    """UI for identifying people in a group photo using reference database.
    
    Args:
        api_url: Base URL for the API
    """
    st.subheader("Identify People in Group Photo")
    
    # Form instructions
    st.write("""
    Upload a group photo to identify people from your reference database.
    """)
    
    # Form inputs
    img_file = st.file_uploader("Group photo", type=SUPPORTED_FORMATS)
    
    # Form validation
    if not img_file or not st.button("Detect & Match"):
        return
    
    # Process identification
    with st.spinner("Identifying faces..."):
        result = identify_faces(img_file, api_url)
        
        if not result:
            st.error("No faces were detected or there was an API error.")
            return
            
        # Display results summary
        st.success(f"Found {len(result)} faces in the image")
        
        # Display results with interactive annotations
        image = Image.open(img_file).convert("RGB")
        
        # Generate face IDs based on labels
        face_ids = {}
        for i, face in enumerate(result):
            face_ids[i] = face.get("label", f"Face #{i+1}")
        
        # Create annotated image
        annotated_img, face_info = create_annotated_image(image, result, face_ids)
        
        # Display the interactive image with JS hover
        st.subheader("Identification Results")
        st.info("Hover over faces to see identification details")
        
        # Create a tooltip formatter function
        def tooltip_formatter(face: Dict) -> str:
            label = face.get("label", "Unknown")
            distance = face.get("distance")
            
            if label != "unknown" and distance is not None:
                confidence = (1 - distance) * 100
                return f"<b>✓ MATCHED: {label}</b><br>Confidence: {confidence:.1f}%"
            else:
                return f"<b>No match found</b><br>This face doesn't match anyone in your reference database."
        
        # Display with JavaScript hover
        display_interactive_image(image, result, tooltip_formatter)
        
        # Also display the annotated image as fallback
        with st.expander("Static Image (with labels)"):
            st.image(annotated_img, caption="Identification Results", use_column_width=True)
        
        # Display detailed information for each face
        st.subheader("Detailed Results")
        st.info("👇 Click on the expanders below to see detailed information for each face")
        
        for face in face_info:
            with st.expander(f"Details for {face['id']}"):
                st.markdown(format_face_attributes(face['data']))
        
        # Create a summary table
        st.subheader("Identification Summary")
        create_face_table(result)
        
        # Display raw JSON data
        with st.expander("View full API response"):
            # Add debug info about reference database
            if result and "debug" in result[0]:
                debug = result[0]["debug"]
                st.subheader("Reference Database Info")
                st.write(f"Reference path: {debug.get('reference_path', 'N/A')}")
                st.write(f"Path exists: {debug.get('reference_exists', False)}")
                st.write(f"Number of reference files: {debug.get('reference_count', 0)}")
                
                if debug.get("reference_contents", []):
                    st.subheader("Sample Reference Files")
                    for i, ref_file in enumerate(debug.get("reference_contents", [])[:10]):
                        st.text(f"{i+1}. {ref_file}")
                
                # Remove debug info to avoid cluttering the response
                result_clean = [result[i].copy() for i in range(len(result))]
                for face in result_clean:
                    if "debug" in face:
                        del face["debug"]
                    
                st.subheader("API Response Data")
                display_json(result_clean)
            else:
                display_json(result)

def identify_within_group_ui(api_url: str) -> None:
    """UI for identifying a specific person in a group photo using a reference photo.
    
    Args:
        api_url: Base URL for the API
    """
    st.subheader("Find Person in Group Photo")
    
    # Form instructions
    st.write("""
    Upload a reference photo of a specific person and a group photo to find them in.
    """)
    
    # Form inputs
    col1, col2 = st.columns(2)
    with col1:
        reference = st.file_uploader("Reference photo of person to find", type=SUPPORTED_FORMATS, key="reference")
    with col2:
        group = st.file_uploader("Group photo to search in", type=SUPPORTED_FORMATS, key="group")
    
    # Form validation
    if not group or not reference or not st.button("Find in Group"):
        return
    
    # Process identification
    with st.spinner("Finding matches..."):
        # Step 1: First show the reference photo
        ref_img = Image.open(reference).convert("RGB")
        st.write("Reference Photo:")
        st.image(ref_img, width=300)
        
        # Step 2: Detect all faces in group photo
        with st.spinner("Detecting faces in group photo..."):
            matches = identify_faces(group, api_url)
            
            if not matches:
                st.error("No faces detected in the group photo or API request failed")
                return
                
            st.info(f"Detected {len(matches)} faces in the group photo")
        
        # Step 3: Process each face against the reference photo
        image = Image.open(group).convert("RGB")
        matching_faces = []
        processed_faces = []
        
        with st.spinner(f"Comparing {len(matches)} faces against reference..."):
            # Create a progress bar
            progress_bar = st.progress(0)
            
            for i, match in enumerate(matches):
                bbox = match["bbox"]
                face_id = f"Face #{i+1}"
                
                # Update the current face being processed
                st.text(f"Processing {face_id}...")
                
                processed_face = {
                    "bbox": bbox,
                    "id": face_id,
                    "verified": False,
                    "distance": 1.0,  # Default to maximum distance (no match)
                    "processing_status": "Processing"
                }
                
                # Crop face region from group photo
                try:
                    # Extract coordinates and crop face
                    crop = crop_face(image, bbox)
                    
                    # Make sure the crop is reasonable size
                    min_size = 50  # Minimum size in pixels
                    if crop.width < min_size or crop.height < min_size:
                        processed_face["processing_status"] = f"Face too small ({crop.width}x{crop.height})"
                        processed_faces.append(processed_face)
                        continue
                    
                    # Convert to bytes for API with high quality to avoid data loss
                    img_byte_arr = io.BytesIO()
                    crop.save(img_byte_arr, format='JPEG', quality=95)
                    img_byte_arr.seek(0)
                    
                    # Get byte data and verify it's not empty
                    byte_data = img_byte_arr.getvalue()
                    if not byte_data or len(byte_data) < 100:  # Very small images are suspicious
                        processed_face["processing_status"] = f"Invalid image data ({len(byte_data)} bytes)"
                        processed_faces.append(processed_face)
                        continue
                    
                    # Prepare API data - using dictionary format for more reliable behavior
                    files = {
                        "img1": ("face.jpg", byte_data, "image/jpeg"),
                        "img2": ("reference.jpg", reference.getvalue(), reference.type)
                    }
                    
                    # Compare directly with API request
                    from utils.api import api_request
                    comp = api_request("compare", files, api_url)
                    
                    if comp:
                        processed_face["distance"] = comp.get("distance", 1.0)
                        processed_face["verified"] = comp.get("verified", False)
                        
                        # Note if face detection was disabled
                        if comp.get("face_detection_enforced") is False:
                            processed_face["processing_status"] = "Compared (face detection relaxed)"
                        else:
                            processed_face["processing_status"] = "Compared"
                        
                        if comp.get("verified", False):
                            matching_faces.append(processed_face)
                    else:
                        processed_face["processing_status"] = "Comparison failed"
                
                except Exception as e:
                    processed_face["processing_status"] = f"Error: {str(e)}"
                    st.error(f"Error processing face #{i+1}: {str(e)}")
                
                # Add to processed faces list
                processed_faces.append(processed_face)
                
                # Update progress
                progress_bar.progress((i + 1) / len(matches))
            
            # Complete progress
            progress_bar.progress(1.0)
        
        # Display results
        if matching_faces:
            st.success(f"Found {len(matching_faces)} matching faces in the group photo")
            
            # Create face IDs dictionary
            face_ids = {}
            for i, face in enumerate(processed_faces):
                face_id = f"Face #{i+1}"
                if face.get("verified", False):
                    confidence = (1 - face.get("distance", 0)) * 100
                    face_ids[i] = f"Match! ({confidence:.1f}%)"
                else:
                    face_ids[i] = face_id
            
            # Create annotated image
            annotated_img, _ = create_annotated_image(image, processed_faces, face_ids)
            
            # Display the interactive image
            st.subheader("Matching Results")
            st.info("Hover over faces to see match details")
            
            # Create tooltip formatter
            def match_tooltip(face: Dict) -> str:
                if face.get("verified", False):
                    confidence = (1 - face.get("distance", 0)) * 100
                    return f"<b>MATCH ✓</b><br>Confidence: {confidence:.1f}%"
                else:
                    return "Not a match"
            
            # Display with JavaScript hover
            display_interactive_image(image, processed_faces, match_tooltip)
            
            # Also show static image as fallback
            with st.expander("Static Image (with labels)"):
                st.image(annotated_img, caption="Green boxes indicate matches", use_column_width=True)
            
            # Display details for matches
            st.subheader("Matching Faces Details")
            for face in matching_faces:
                confidence = (1 - face.get("distance", 0)) * 100
                st.markdown(f"**{face.get('id')}** - Confidence: {confidence:.1f}%")
        else:
            st.warning("No matching faces found in the group photo")
            
            # Still show detected faces
            annotated_img, _ = create_annotated_image(image, processed_faces)
            
            # Display the image
            st.image(annotated_img, caption="No matching faces found", use_column_width=True)
        
        # Display detailed results
        with st.expander("View detection details"):
            st.subheader("Processed Faces")
            for face in processed_faces:
                status = "✅ Match" if face.get("verified", False) else "❌ No match"
                confidence = (1 - face.get("distance", 1.0)) * 100
                st.markdown(f"**{face.get('id')}**: {status} - Confidence: {confidence:.1f}% - {face.get('processing_status', '')}")
            
            st.subheader("Raw Data")
            display_json(processed_faces)
