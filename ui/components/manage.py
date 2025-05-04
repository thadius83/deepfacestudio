"""Reference database management component for DeepFace Suite."""

import streamlit as st
from typing import Dict, List, Optional, Any
import time
import requests

from utils.ui import display_json, SUPPORTED_FORMATS

# -------- helpers for confirmation dialogs --------
def _confirm_key(action: str) -> str:
    """Generate a unique session state key for confirmation actions."""
    return f"confirm_{action}"

def _want_confirmation(action: str) -> bool:
    """Check if confirmation is needed for the given action."""
    return st.session_state.get(_confirm_key(action), False)

def _set_confirmation(action: str, value: bool = True):
    """Set confirmation state for the given action."""
    st.session_state[_confirm_key(action)] = value

def manage_references_ui(api_url: str) -> None:
    """UI for managing reference photos database.
    
    Args:
        api_url: Base URL for the API
    """
    st.subheader("Manage Reference Database")
    
    # Fetch reference data
    with st.spinner("Loading reference database..."):
        try:
            response = requests.get(f"{api_url}/reference")
            if response.status_code == 200:
                ref_data = response.json()
                if "error" in ref_data:
                    st.error(f"Error loading references: {ref_data['error']}")
                    return
            else:
                st.error(f"Failed to retrieve reference data: {response.status_code}")
                return
        except Exception as e:
            st.error(f"Failed to connect to API: {str(e)}")
            return
    
    # Calculate database stats
    labels = ref_data.get("labels", {})
    total_labels = len(labels)
    total_photos = sum(label_data.get("count", 0) for label_data in labels.values())
    
    # Display database summary
    st.write(f"Found **{total_labels}** labels with **{total_photos}** total reference photos.")
    
    if total_labels == 0:
        st.warning("No reference photos found. Use the 'Add reference photos' option to add some.")
        return
    
    # Add refresh button
    if st.button("🔄 Refresh Database"):
        st.rerun()
    
    # Display each label with its photos
    for label, label_data in labels.items():
        with st.expander(f"📁 {label} ({label_data.get('count', 0)} photos)"):
            # Add a delete label button with proper session state management
            action_id = f"delete_label_{label}"
            
            if st.button(f"🗑️ Delete All Photos for '{label}'", key=action_id):
                _set_confirmation(action_id, True)
            
            if _want_confirmation(action_id):
                st.warning(f"⚠️ Really delete **{label}** and all its photos?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Cancel", key=f"cancel_{action_id}"):
                        _set_confirmation(action_id, False)
                with col2:
                    if st.button("✓ Confirm Delete", key=f"do_{action_id}", type="primary"):
                        _set_confirmation(action_id, False)  # reset the flag
                        with st.spinner(f"Deleting label '{label}'..."):
                            r = requests.delete(f"{api_url}/reference/{label}", timeout=30)
                        if r.ok:
                            st.success(f"Successfully deleted label '{label}'")
                            st.rerun()
                        else:
                            st.error(f"Failed to delete label: {r.status_code}")
                            st.error(f"Response text: {r.text}")
            
            # Display photos in a grid
            files = label_data.get("files", [])
            cols = st.columns(4)  # Show 4 photos per row
            
            for i, file in enumerate(files):
                with cols[i % 4]:
                    file_id = file.get("id", "")
                    file_path = file.get("path", "")
                    file_name = file.get("filename", "")
                    
                    # Display thumbnail image
                    # We need to proxy the image through our API utility to avoid cross-origin issues
                    try:
                        # Use a placeholder to show a loading spinner
                        with st.spinner("Loading image..."):
                            # Fetch the image data through our API
                            image_response = requests.get(f"{api_url}/reference/image/{label}/{file_id}")
                            if image_response.status_code == 200:
                                # Use BytesIO to convert the response content to an image
                                import io
                                from PIL import Image
                                img = Image.open(io.BytesIO(image_response.content))
                                st.image(img, caption=f"{label}", width=150)
                            else:
                                raise Exception(f"Failed to load image: {image_response.status_code}")
                    except Exception as e:
                        st.warning(f"Image preview unavailable: {str(e)}")
                        
                        # Show alternative placeholder
                        st.markdown(f"""
                        <div style="width:150px;height:150px;background:#f0f0f0;
                        display:flex;align-items:center;justify-content:center;
                        border-radius:5px;color:#666;font-size:12px;text-align:center;">
                        {label}<br>ID: {file_id[:8]}...
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display photo identifier
                    st.text(f"ID: {file_id[:8]}...")
                    
                    # Add delete button for individual photo with session state pattern
                    photo_action_id = f"delete_photo_{file_id}"
                    
                    if st.button("🗑️ Delete", key=photo_action_id):
                        _set_confirmation(photo_action_id, True)
                    
                    if _want_confirmation(photo_action_id):
                        st.warning(f"Confirm deletion of this photo?")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Cancel", key=f"cancel_{photo_action_id}"):
                                _set_confirmation(photo_action_id, False)
                        with col2:
                            if st.button("✓ Confirm Delete", key=f"do_{photo_action_id}", type="primary"):
                                _set_confirmation(photo_action_id, False)  # reset the flag
                                with st.spinner(f"Deleting photo..."):
                                    r = requests.delete(f"{api_url}/reference/{label}/{file_id}", timeout=30)
                                if r.ok:
                                    st.success(f"Deleted photo {file_id[:8]}...")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to delete photo: {r.status_code}")
                                    st.error(f"Response: {r.text}")
    
    # Administration section
    st.subheader("Database Administration")
    
    # Option to delete entire database - using the session state pattern
    st.markdown("---")
    st.subheader("Reset Database")
    st.warning("⚠️ WARNING: This will delete ALL reference photos and reset the database")
    
    reset_action = "reset_db"
    
    if st.button("Reset Entire Database", key=reset_action, type="primary"):
        _set_confirmation(reset_action)
    
    if _want_confirmation(reset_action):
        st.error("⚠️ This will erase **all** reference photos.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Cancel", key=f"cancel_{reset_action}"):
                _set_confirmation(reset_action, False)
        with col2:
            if st.button("✓ CONFIRM COMPLETE RESET", key=f"do_{reset_action}", type="primary"):
                _set_confirmation(reset_action, False)  # reset the flag
                with st.spinner("Resetting database..."):
                    r = requests.delete(f"{api_url}/reference/all", timeout=60)
                
                # Display response details (useful for debugging)
                st.info(f"Response status: {r.status_code}")
                try:
                    response_data = r.json()
                    st.info(f"Response: {response_data}")
                except:
                    st.info(f"Response text: {r.text}")
                
                # Process the response
                if r.ok:
                    st.success("Database successfully reset")
                    
                    # Force a database rebuild
                    st.write("Rebuilding database representations...")
                    rebuild_url = f"{api_url}/reference/rebuild"
                    rebuild_response = requests.post(rebuild_url)
                    
                    if rebuild_response.status_code == 200:
                        st.success("Reference database representations rebuilt")
                    else:
                        st.warning(f"Database reset successful but rebuild failed: {rebuild_response.status_code}")
                    
                    # Refresh the page
                    st.rerun()
                else:
                    st.error(f"Failed to reset database: {r.status_code}")
                    st.error(f"Error details: {r.text}")
