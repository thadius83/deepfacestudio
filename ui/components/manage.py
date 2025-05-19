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
    
    # Add search functionality
    search_term = st.text_input("🔍 Search by label name", key="search_labels")
    
    # Items per page selector
    col1, col2 = st.columns([2, 1])
    with col1:
        items_per_page_options = [10, 20, 50]
        items_per_page = st.selectbox(
            "Items per page:", 
            options=items_per_page_options,
            index=0,
            key="items_per_page"
        )
    
    with col2:
        # Add refresh button
        if st.button("🔄 Refresh Database", key="refresh_database_button"):
            st.rerun()
    
    # Filter labels based on search term
    filtered_labels = {k: v for k, v in labels.items() 
                      if search_term.lower() in k.lower()}
    
    # Pagination logic
    total_filtered = len(filtered_labels)
    total_pages = max(1, (total_filtered + items_per_page - 1) // items_per_page)
    
    # Current page (stored in session state)
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1
    
    # Ensure current page is valid after filtering
    if st.session_state.current_page > total_pages:
        st.session_state.current_page = total_pages
    
    # Pagination controls
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("◀ Previous", disabled=(st.session_state.current_page <= 1), key="prev_page"):
            st.session_state.current_page -= 1
    with col3:
        if st.button("Next ▶", disabled=(st.session_state.current_page >= total_pages), key="next_page"):
            st.session_state.current_page += 1
    with col2:
        st.write(f"Page {st.session_state.current_page} of {total_pages} (Showing {total_filtered} of {total_labels} labels)")
    
    # Get current page entries
    start_idx = (st.session_state.current_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_filtered)
    current_page_labels = dict(list(filtered_labels.items())[start_idx:end_idx])
    
    # Initialize selected photos session state if not exists
    if "selected_photos" not in st.session_state:
        st.session_state.selected_photos = {}
    
    # Display only the current page of labels
    for label, label_data in current_page_labels.items():
        with st.expander(f"📁 {label} ({label_data.get('count', 0)} photos)"):
            # Add a delete label button with proper session state management
            action_id = f"delete_label_{label}"
            
            # Initialize selected photos for this label if not exists
            if label not in st.session_state.selected_photos:
                st.session_state.selected_photos[label] = set()
            
            # Delete actions row
            col1, col2 = st.columns([2, 1])
            with col1:
                if st.button(f"🗑️ Delete All Photos for '{label}'", key=action_id):
                    _set_confirmation(action_id, True)
            
            # Add "Delete Selected" button only if there are photos
            if label_data.get("files", []):
                with col2:
                    selected_count = len(st.session_state.selected_photos[label])
                    delete_selected_action_id = f"delete_selected_{label}"
                    
                    if selected_count > 0:
                        if st.button(f"🗑️ Delete Selected ({selected_count})", key=delete_selected_action_id):
                            _set_confirmation(delete_selected_action_id, True)
            
            # Confirmation for delete all
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
                            # Clear selected photos for this label
                            st.session_state.selected_photos[label] = set()
                            st.success(f"Successfully deleted label '{label}'")
                            st.rerun()
                        else:
                            st.error(f"Failed to delete label: {r.status_code}")
                            st.error(f"Response text: {r.text}")
            
            # Confirmation for delete selected
            if label_data.get("files", []) and _want_confirmation(f"delete_selected_{label}"):
                selected_count = len(st.session_state.selected_photos[label])
                st.warning(f"⚠️ Really delete **{selected_count}** selected photos for '{label}'?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Cancel", key=f"cancel_selected_{label}"):
                        _set_confirmation(f"delete_selected_{label}", False)
                with col2:
                    if st.button("✓ Confirm Delete Selected", key=f"do_selected_{label}", type="primary"):
                        _set_confirmation(f"delete_selected_{label}", False)  # reset the flag
                        
                        # Use batch delete endpoint instead of individual delete requests
                        with st.spinner(f"Deleting {selected_count} selected photos..."):
                            try:
                                # Convert set of file_ids to list
                                file_ids_list = list(st.session_state.selected_photos[label])
                                
                                # Use the batch delete endpoint
                                response = requests.post(
                                    f"{api_url}/reference/{label}/batch-delete",
                                    json=file_ids_list,
                                    timeout=60
                                )
                                
                                # Process the response
                                if response.status_code == 200:
                                    result = response.json()
                                    deleted_count = result.get("total_deleted", 0)
                                    
                                    if deleted_count > 0:
                                        st.success(f"Successfully deleted {deleted_count} photos")
                                        # Clear selected photos for this label
                                        st.session_state.selected_photos[label] = set()
                                        
                                    # Show any errors that occurred
                                    errors = result.get("errors", [])
                                    if errors:
                                        st.error("Some photos could not be deleted:")
                                        for error in errors:
                                            st.error(f"File ID {error.get('file_id')}: {error.get('error')}")
                                    
                                    # Only rerun if we had successful deletions
                                    if deleted_count > 0:
                                        st.rerun()
                                else:
                                    st.error(f"Batch delete failed: {response.status_code}")
                                    st.error(f"Error: {response.text}")
                            except Exception as e:
                                st.error(f"Error during batch delete: {str(e)}")
            
            # Display photos in a grid with selection checkboxes
            files = label_data.get("files", [])
            cols = st.columns(4)  # Show 4 photos per row
            
            # Add "Select All" / "Deselect All" controls
            if files:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Select All", key=f"select_all_{label}_{st.session_state.current_page}"):
                        # Add all file IDs for this label to selected set
                        for file in files:
                            st.session_state.selected_photos[label].add(file.get("id", ""))
                        st.rerun()
                with col2:
                    if st.button("Deselect All", key=f"deselect_all_{label}_{st.session_state.current_page}"):
                        # Clear selected photos for this label
                        st.session_state.selected_photos[label] = set()
                        st.rerun()
            
            for i, file in enumerate(files):
                with cols[i % 4]:
                    file_id = file.get("id", "")
                    file_path = file.get("path", "")
                    file_name = file.get("filename", "")
                    
                    # Add selection checkbox
                    checkbox_key = f"select_{st.session_state.current_page}_{label}_{i}_{file_id[:8]}"
                    is_selected = file_id in st.session_state.selected_photos[label]
                    
                    if st.checkbox("Select", key=checkbox_key, value=is_selected):
                        # Add to selected set
                        st.session_state.selected_photos[label].add(file_id)
                    else:
                        # Remove from selected set if present
                        if file_id in st.session_state.selected_photos[label]:
                            st.session_state.selected_photos[label].remove(file_id)
                    
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
                    # Use a completely unique key by including page number and item index to prevent collisions
                    photo_action_id = f"delete_photo_{st.session_state.current_page}_{label}_{i}_{file_id[:8]}"
                    
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
                                    # Remove from selected set if present
                                    if file_id in st.session_state.selected_photos[label]:
                                        st.session_state.selected_photos[label].remove(file_id)
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
