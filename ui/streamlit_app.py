"""
Streamlit UI for DeepFace suite.
Run by `streamlit run streamlit_app.py`.
"""

import streamlit as st
import sys
import os
from ui_config import API_URL

# Add directories to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)  # Add parent directory to path

# Import component modules
from components.reference import reference_upload_ui
from components.identify import identify_group_ui, identify_within_group_ui
from components.compare import compare_photos_ui
from components.analyze import analyze_attributes_ui
from components.manage import manage_references_ui
from components.family_resemblance import family_resemblance_ui

# App configuration
st.set_page_config(
    page_title="DeepFace Suite", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to reduce space above title
st.markdown("""
    <style>
        .main > div {
            padding-top: 1rem;
        }
        .block-container {
            padding-top: 0rem;
            margin-top: 0rem;
        }
        h1 {
            margin-top: 0rem !important;
        }
    </style>
""", unsafe_allow_html=True)

def main():
    """Main application entry point."""
    st.title("🔍 DeepFace Suite – Face Recognition & Analysis")
    
    # Task selection sidebar
    task = st.sidebar.radio(
        "Select task",
        (
            "Add reference photos", 
            "Manage reference database",
            "Identify in group photo", 
            "Compare two photos", 
            "Find person in group photo",
            "Analyze attributes",
            "Which Parent Do You Look Like"
        )
    )
    
    # Display contextual help in sidebar
    st.sidebar.write("---")
    st.sidebar.info(
        "This tool uses deep learning for facial recognition and analysis. "
        "Select a task from the options above to get started."
    )
    
    # Add image help text
    st.sidebar.write("---")
    st.sidebar.info(
        "**Interactive Images:** \n"
        "- Hover over faces to see detailed information\n"
        "- Green boxes indicate verified matches\n"
        "- Yellow boxes indicate detected faces\n"
        "- Click on the expanders below images to see additional details"
    )
    
    # Display version information
    st.sidebar.write("---")
    st.sidebar.caption("DeepFace Suite v1.0")
    
    # Route to appropriate task handler
    if task == "Add reference photos":
        reference_upload_ui(API_URL)
    elif task == "Manage reference database":
        manage_references_ui(API_URL)
    elif task == "Identify in group photo":
        identify_group_ui(API_URL)
    elif task == "Compare two photos":
        compare_photos_ui(API_URL)
    elif task == "Find person in group photo":
        identify_within_group_ui(API_URL)
    elif task == "Analyze attributes":
        analyze_attributes_ui(API_URL)
    elif task == "Which Parent Do You Look Like":
        family_resemblance_ui(API_URL)
    else:
        st.error("Unknown task selected. Please choose a valid task from the sidebar.")

if __name__ == "__main__":
    main()
