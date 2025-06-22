from flask import Flask, request, jsonify
import os
import uuid
from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS, API_CONFIG, RESPONSE_MESSAGES
from utils import allowed_file, save_uploaded_file, cleanup_temp_files
from FaceVerification import FaceVerifier
from ocr_passport import process_passport
import logging
from LiveVideo import LiveVideo

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/verify_faces', methods=['POST'])
def verify_faces():
    """
    Endpoint to verify if multiple images belong to the same person
    Expected payload: {'images': [image1, image2, ...]}
    Returns JSON with verification result and confidence
    """
    if 'images' not in request.files:
        return jsonify({
            'success': False,
            'error': RESPONSE_MESSAGES['MISSING_FILE'].format('images')
        }), 400

    files = request.files.getlist('images')
    if len(files) < 2:
        return jsonify({
            'success': False,
            'error': 'At least two images are required for verification'
        }), 400

    saved_paths = []
    try:
        # Save all uploaded images
        for file in files:
            if file.filename == '' or not allowed_file(file.filename):
                continue
            filename = f"{uuid.uuid4().hex}.{file.filename.rsplit('.', 1)[1].lower()}"
            file_path = save_uploaded_file(file, filename)
            saved_paths.append(file_path)

        # Verify faces
        verifier = FaceVerifier()
        is_same_person = verifier.verify_faces(saved_paths)
        
        return jsonify({
            'result': is_same_person,
        })
    except Exception as e:
        logger.error(f"Face verification error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': RESPONSE_MESSAGES['FACE_MATCH_FAILED'].format(str(e))
        }), 500
    finally:
        # Clean up uploaded files
        cleanup_temp_files(saved_paths)

@app.route('/extract_passport', methods=['POST'])
def extract_passport():
    """
    Endpoint to extract MRZ information from a passport image
    Expected payload: {'passport_image': image_file}
    Returns JSON with extracted passport data
    """
    if 'passport_image' not in request.files:
        return jsonify({
            'success': False,
            'error': RESPONSE_MESSAGES['MISSING_FILE'].format('passport_image')
        }), 400

    file = request.files['passport_image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': 'Invalid file'
        }), 400

    try:
        # Save uploaded file
        filename = f"{uuid.uuid4().hex}.{file.filename.rsplit('.', 1)[1].lower()}"
        file_path = save_uploaded_file(file, filename)
        
        # Process passport
        result = process_passport(file_path)
        
        if result['status'] != 'SUCCESS':
            return jsonify({
                'success': False,
                'error': result['message']
            }), 400
            
        return jsonify({
            'success': True,
            'passport_data': result['mrz_info']
        })
    except Exception as e:
        logger.error(f"Passport extraction error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': RESPONSE_MESSAGES['EXTRACTION_ERROR'].format(str(e))
        }), 500
    finally:
        # Clean up uploaded file
        if 'file_path' in locals():
            cleanup_temp_files([file_path])

# Add after existing routes
@app.route('/check_liveness', methods=['POST'])
def check_liveness():
    """
    Endpoint to verify liveness from a video file
    Expected payload: {'video': video_file}
    Returns JSON with liveness result
    """
    if 'video' not in request.files:
        return jsonify({
            'success': False,
            'error': RESPONSE_MESSAGES['MISSING_FILE'].format('video')
        }), 400

    file = request.files['video']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': 'Invalid video file'
        }), 400

    try:
        # Save uploaded file
        filename = f"{uuid.uuid4().hex}.{file.filename.rsplit('.', 1)[1].lower()}"
        file_path = save_uploaded_file(file, filename)
        
        # Process video
        lv = LiveVideo()
        is_live = lv.detect_head_movement(file_path)
        
        return jsonify({
            'success': True,
            'is_live': is_live
        })
    except Exception as e:
        logger.error(f"Liveness check error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': RESPONSE_MESSAGES['LIVENESS_FAILED'] + f": {str(e)}"
        }), 500
    finally:
        # Clean up uploaded file
        if 'file_path' in locals():
            cleanup_temp_files([file_path])

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)