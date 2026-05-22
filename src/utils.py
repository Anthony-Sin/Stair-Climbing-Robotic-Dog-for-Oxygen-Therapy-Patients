import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from io import BytesIO
from PIL import Image

def create_vertical_colorbar(height=480, min_depth=0.0, max_depth=5.0, cmap='jet'):
    """
    Creates a vertical color bar using matplotlib and returns an OpenCV image.
    
    Args:
        height (int): Height of the output colorbar image in pixels.
        min_depth (float): Minimum depth value (e.g. meters).
        max_depth (float): Maximum depth value (e.g. meters).
        cmap (str): Colormap name.
    
    Returns:
        np.ndarray: OpenCV BGR image of the colorbar.
    """
    fig, ax = plt.subplots(figsize=(0.6, 4.8), dpi=100)  # Vertical figure
    norm = plt.Normalize(vmin=min_depth, vmax=max_depth)
    fig.subplots_adjust(left=0.3, right=0.7)
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax)
    cb.set_label('Distance (m)', fontsize=8)
    cb.ax.tick_params(labelsize=8)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    pil_img = Image.open(buf).convert('RGB')
    open_cv_image = np.array(pil_img)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

    # Resize to desired height keeping aspect ratio
    width = int(open_cv_image.shape[1] * height / open_cv_image.shape[0])
    open_cv_image = cv2.resize(open_cv_image, (width, height))

    return open_cv_image

def draw_fps(image, fps, position=(10, 30), color=(0, 255, 0), scale=1, thickness=2, label='FPS'):
    """
    Draw FPS text on an image.
    
    Args:
        image (np.ndarray): The image to draw on.
        fps (float): Frames per second value to display.
        position (tuple): (x, y) position for the text.
        color (tuple): Text color in BGR.
        scale (float): Font scale.
        thickness (int): Font thickness.
    
    Returns:
        None
    """
    text = f"{label}: {fps:.1f}"
    x, y = position
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad = 6
    top_left = (max(0, x - pad), max(0, y - text_h - pad))
    bottom_right = (
        min(image.shape[1] - 1, x + text_w + pad),
        min(image.shape[0] - 1, y + baseline + pad),
    )
    cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def rotate_image(image, angle):
    """Rotate image clockwise by angle degrees."""
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError("Only 90, 180, 270 degrees supported.")
