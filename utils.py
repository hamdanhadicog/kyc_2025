import os
import logging
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS
import datetime
import imghdr
import mimetypes
from pdf2image import convert_from_path
import glob

logger = logging.getLogger(__name__)
def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(file, filename=None):
    """Save an uploaded file securely."""
    if filename is None:
        filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        file.save(file_path)
        return file_path
    except Exception as e:
        logger.error(f"Error saving file {filename}: {str(e)}")
        raise

def save_debug_image(image, filename):
    """Save a debug image if it's valid."""
    if image is not None and isinstance(image, np.ndarray) and image.shape[-1] == 3:
        try:
            debug_path = os.path.join(UPLOAD_FOLDER, filename)
            cv2.imwrite(debug_path, image)
            return debug_path
        except Exception as e:
            logger.error(f"Error saving debug image {filename}: {str(e)}")
    return None

def normalize_image(image):
    """Normalize image to uint8 format."""
    if image is None:
        return None
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
    return image

def create_response(success, data=None, error=None, status_code=200):
    """Create a standardized API response."""
    response = {
        'success': success,
        'timestamp': datetime.datetime.now(datetime.UTC).isoformat()
    }
    if data is not None:
        response['data'] = data
    if error is not None:
        response['error'] = error
    return response, status_code

def is_image_file(filepath):
    """Check if file is a valid image (jpeg, png, or pdf)."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.pdf':
        return True
    return imghdr.what(filepath) in ['jpeg', 'png','jpg']

def is_video_file(filepath):
    """Check if file is a valid video by MIME type and frame count."""
    mime = mimetypes.guess_type(filepath)[0]
    if mime and mime.startswith('video'):
        # Extra check: try to open with OpenCV
        try:
            cap = cv2.VideoCapture(filepath)
            ret, _ = cap.read()
            cap.release()
            return ret  # True if at least one frame can be read
        except Exception:
            return False
    return False

def rotate_image(image, angle=0):
    """
    Rotate an image by the given angle (in degrees).
    angle: 0, 90, 180, 270
    """
    if angle == 0:
        return image
    elif angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError("Angle must be 0, 90, 180, or 270")

def pdf_to_image(pdf_path):
    """
    Convert the first page of a PDF to a JPEG image.
    Returns the path to the saved image.
    """
    images = convert_from_path(pdf_path, first_page=1, last_page=1)
    if not images:
        return None
    image_path = pdf_path + "_page1.jpg"
    images[0].save(image_path, 'JPEG')
    return image_path

def is_valid_pdf(filepath):
    """
    Check if file is a valid PDF by examining its header.
    
    Args:
        filepath: Path to the file to check
        
    Returns:
        bool: True if file is a valid PDF, False otherwise
    """
    try:
        with open(filepath, 'rb') as f:
            header = f.read(5)
            return header.startswith(b'%PDF-')
    except Exception as e:
        logger.error(f"Error validating PDF file {filepath}: {str(e)}")
        return False

# utils.py (partial update)
def cleanup_temp_files(file_paths):
    try:
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)
            
            # Clean up temporary face images from video processing
            if "temp_face_" in path:
                os.remove(path)
                
        # Clean up all temporary face images
        for temp_face in glob.glob("temp_face_*.jpg"):
            os.remove(temp_face)
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

def cleanup_preprocessed_images(pattern="preprocessed_mrz_*.jpg"):
    """Delete all preprocessed MRZ images matching the pattern."""
    for file in glob.glob(pattern):
        try:
            os.remove(file)
            logger.info(f"Deleted preprocessed image: {file}")
        except Exception as e:
            logger.warning(f"Could not delete {file}: {e}")