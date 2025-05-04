/**
 * JavaScript functions for face hover interactions
 */

// Show tooltip when hovering over a face box
function showTooltip(event, id) {
    const tooltip = document.getElementById('tooltip-' + id);
    if (tooltip) {
        tooltip.style.display = 'block';
        tooltip.style.left = (event.clientX + 10) + 'px';
        tooltip.style.top = (event.clientY + 10) + 'px';
    }
}

// Hide tooltip when mouse leaves face box
function hideTooltip(id) {
    const tooltip = document.getElementById('tooltip-' + id);
    if (tooltip) {
        tooltip.style.display = 'none';
    }
}

// Update tooltip position when mouse moves over face box
function updateTooltipPosition(event, id) {
    const tooltip = document.getElementById('tooltip-' + id);
    if (tooltip && tooltip.style.display === 'block') {
        tooltip.style.left = (event.clientX + 10) + 'px';
        tooltip.style.top = (event.clientY + 10) + 'px';
    }
}

// Initialize all face hover elements
function initFaceHover() {
    document.querySelectorAll('.face-box').forEach(box => {
        const id = box.getAttribute('data-face-id');
        
        box.addEventListener('mouseenter', function(event) {
            showTooltip(event, id);
        });
        
        box.addEventListener('mouseleave', function() {
            hideTooltip(id);
        });
        
        box.addEventListener('mousemove', function(event) {
            updateTooltipPosition(event, id);
        });
    });
}

// Run initialization when document is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initFaceHover);
} else {
    // Document already loaded
    initFaceHover();
}
