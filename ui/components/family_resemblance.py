"""Family resemblance component for DeepFace Suite."""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

from utils.api import family_resemblance
from utils.image import create_annotated_image, display_interactive_image
from utils.ui import display_json, format_face_attributes, SUPPORTED_FORMATS

def family_resemblance_ui(api_url: str) -> None:
    """UI for determining which parent a child resembles more.
    
    Args:
        api_url: Base URL for the API
    """
    st.subheader("Family Resemblance")
    
    # Form instructions
    st.write("""
    Upload photos of a father, mother, and child to see which parent the child resembles more.
    The app will analyze facial features and determine resemblance based on facial similarity.
    """)
    
    # Form inputs in three columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Father**")
        father_img = st.file_uploader("Upload father's photo", type=SUPPORTED_FORMATS, key="father_img")
    with col2:
        st.write("**Child**")
        child_img = st.file_uploader("Upload child's photo", type=SUPPORTED_FORMATS, key="child_img")
    with col3:
        st.write("**Mother**")
        mother_img = st.file_uploader("Upload mother's photo", type=SUPPORTED_FORMATS, key="mother_img")
    
    # Form validation
    if not father_img or not mother_img or not child_img or not st.button("Analyze Resemblance", key="analyze_resemblance_button"):
        return
    
    # Process comparison
    with st.spinner("Analyzing family resemblance..."):
        # Use the family resemblance API endpoint
        result = family_resemblance(father_img, child_img, mother_img, api_url)
        
        if not result:
            st.error("Could not compare faces. Check if all images contain detectable faces.")
            return
        
        # Get distance metrics
        paternal_result = result.get("paternal_result", {})
        maternal_result = result.get("maternal_result", {})
        paternal_distance = paternal_result.get("distance", 999)
        maternal_distance = maternal_result.get("distance", 999)
        
        # Display family photos with matplotlib
        display_family_comparison(father_img, child_img, mother_img, 
                                 paternal_distance, maternal_distance)
        
        # Show results
        st.subheader("Resemblance Results")
        
        # Create metrics
        col1, col2 = st.columns(2)
        with col1:
            # Lower distance means more similarity
            father_similarity = max(0, 100 - (paternal_distance * 5))  # Convert to a similarity percentage
            st.metric("Father Similarity", f"{father_similarity:.1f}%")
        
        with col2:
            mother_similarity = max(0, 100 - (maternal_distance * 5))  # Convert to a similarity percentage
            st.metric("Mother Similarity", f"{mother_similarity:.1f}%")
        
        # Show conclusion
        st.subheader("Conclusion")
        resemblance = result.get("resemblance", "unknown")
        if resemblance == "father":
            st.success("The child looks more like their father.")
        elif resemblance == "mother":
            st.success("The child looks more like their mother.")
        else:
            st.warning("Could not determine resemblance.")
        
        # Display technical details
        with st.expander("View technical details"):
            st.write("**Paternal Comparison**")
            display_json(paternal_result)
            st.write("**Maternal Comparison**")
            display_json(maternal_result)

def display_family_comparison(father_img, child_img, mother_img, paternal_distance, maternal_distance):
    """Create and display family comparison visualization similar to the reference image.
    
    Args:
        father_img: Father's image file
        child_img: Child's image file
        mother_img: Mother's image file
        paternal_distance: Distance metric between father and child
        maternal_distance: Distance metric between mother and child
    """
    # Open images with PIL
    father = Image.open(father_img).convert("RGB")
    child = Image.open(child_img).convert("RGB")
    mother = Image.open(mother_img).convert("RGB")
    
    # Determine which parent is more similar
    paternal_match = paternal_distance < maternal_distance
    
    # Create a single-row figure
    fig = plt.figure(figsize=(16, 6))
    
    # Create a layout with appropriate spacing
    grid = plt.GridSpec(1, 5, width_ratios=[1, 0.5, 1, 0.5, 1])
    
    # Add the images as circular thumbnails
    # Father image
    ax1 = plt.subplot(grid[0])
    circle1 = plt.Circle((0.5, 0.5), 0.5, transform=ax1.transAxes, edgecolor='gray', facecolor='none', linewidth=3)
    ax1.add_patch(circle1)
    ax1.imshow(crop_to_circle(father))
    ax1.set_title("Father", fontsize=14, pad=20)
    ax1.axis("off")
    
    # Child image
    ax3 = plt.subplot(grid[2])
    circle3 = plt.Circle((0.5, 0.5), 0.5, transform=ax3.transAxes, edgecolor='gray', facecolor='none', linewidth=3)
    ax3.add_patch(circle3)
    ax3.imshow(crop_to_circle(child))
    ax3.set_title("Child", fontsize=14, pad=20)
    ax3.axis("off")
    
    # Mother image
    ax5 = plt.subplot(grid[4])
    circle5 = plt.Circle((0.5, 0.5), 0.5, transform=ax5.transAxes, edgecolor='gray', facecolor='none', linewidth=3)
    ax5.add_patch(circle5)
    ax5.imshow(crop_to_circle(mother))
    ax5.set_title("Mother", fontsize=14, pad=20)
    ax5.axis("off")
    
    # Add the connecting lines and distances
    # Father to Child connection
    ax2 = plt.subplot(grid[1])
    ax2.axis("off")
    ax2.text(0.5, 0.6, f"{paternal_distance:.2f}", fontsize=16, ha='center')
    
    # Draw horizontal arrow
    ax2.annotate("", xy=(0, 0.5), xytext=(1, 0.5),
                arrowprops=dict(arrowstyle='<->', color='black', linewidth=2))
    
    # Add the red X or green check mark
    if paternal_match:
        # Green check if father is more similar
        ax2.text(0.5, 0.3, "✓", fontsize=40, ha='center', va='center', color='green', 
                weight='bold', bbox=dict(facecolor='white', edgecolor='green', boxstyle='circle'))
    else:
        # Red X if father is less similar
        ax2.text(0.5, 0.3, "✗", fontsize=40, ha='center', va='center', color='red', 
                weight='bold', bbox=dict(facecolor='white', edgecolor='red', boxstyle='circle'))
    
    # Child to Mother connection
    ax4 = plt.subplot(grid[3])
    ax4.axis("off")
    ax4.text(0.5, 0.6, f"{maternal_distance:.2f}", fontsize=16, ha='center')
    
    # Draw horizontal arrow
    ax4.annotate("", xy=(0, 0.5), xytext=(1, 0.5),
                arrowprops=dict(arrowstyle='<->', color='black', linewidth=2))
    
    # Add the red X or green check mark
    if not paternal_match:
        # Green check if mother is more similar
        ax4.text(0.5, 0.3, "✓", fontsize=40, ha='center', va='center', color='green', 
                weight='bold', bbox=dict(facecolor='white', edgecolor='green', boxstyle='circle'))
    else:
        # Red X if mother is less similar
        ax4.text(0.5, 0.3, "✗", fontsize=40, ha='center', va='center', color='red', 
                weight='bold', bbox=dict(facecolor='white', edgecolor='red', boxstyle='circle'))
    
    # Add the result at the bottom
    plt.figtext(0.5, 0.05, 
               f"The child looks more like their {'father' if paternal_match else 'mother'}.",
               ha='center', fontsize=16, 
               bbox=dict(facecolor='lightblue', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Save figure to a buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    # Display in Streamlit
    st.image(buf, use_column_width=True)
    
    # Close the figure to free memory
    plt.close(fig)

def crop_to_circle(img):
    """Crop an image to a circle for better visualization.
    
    Args:
        img: PIL Image
        
    Returns:
        numpy array of the circular-cropped image
    """
    # Convert to numpy array
    img_array = np.array(img)
    
    # Get dimensions
    h, w = img_array.shape[:2]
    
    # Create a circular mask
    radius = min(h, w) // 2
    center = (w // 2, h // 2)
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    
    # Apply mask
    mask = dist_from_center <= radius
    masked_img = img_array.copy()
    
    # Make outside of circle transparent
    for i in range(3):  # RGB channels
        masked_img[:, :, i] = np.where(mask, masked_img[:, :, i], 255)
    
    return masked_img
