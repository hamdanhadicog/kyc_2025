from flask import Flask, request, jsonify
import os
import uuid
from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS, API_CONFIG, RESPONSE_MESSAGES
from utils import allowed_file, save_uploaded_file, cleanup_temp_files
from FaceVerification import FaceVerifier
from ocr_passport import process_passport
import logging
from LiveVideo import LiveVideo
from ocr_id import OcrId  # Make sure this matches the filename and class
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["1 per minute"]  # Adjust based on your needs
)

app.rate_limit_exceeded_handler = lambda e: jsonify({
    "success": False,
    "error": "Rate limit exceeded. Try again in {} seconds.".format(int(e.description)),
    "retry_after_seconds": int(e.description)
}), 429


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

verifier = FaceVerifier()




@app.route('/verify_faces', methods=['POST'])
@limiter.limit("3 per minute")  # Stricter limit for this heavy endpoint
def verify_faces():
    """
    Endpoint to verify if ID, Passport, Selfie are of the same person,
    and also check if the provided video is live.

    Expected form-data fields:
        - id_image
        - passport_image
        - selfie_image
        - video
    """

    # Define required fields
    required_files = ['id_image', 'passport_image', 'selfie_image', 'video']
    received_files = request.files

    # Validate presence of all required files
    missing_files = [field for field in required_files if field not in received_files]
    if missing_files:
        return jsonify({
            'success': False,
            'error': RESPONSE_MESSAGES['MISSING_FILE'].format(', '.join(missing_files))
        }), 400

    saved_paths = {}
    try:
        # Save each file individually
        for field_name in required_files:
            file = received_files[field_name]
            if file.filename == '' or not allowed_file(file.filename):
                return jsonify({
                    'success': False,
                    'error': f"Invalid file for {field_name}"
                }), 400

            ext = file.filename.rsplit('.', 1)[1].lower()
            filename = f"{uuid.uuid4().hex}.{ext}"
            file_path = save_uploaded_file(file, filename)
            saved_paths[field_name] = file_path

        # Step 1: Face Verification
        face_result = verifier.verify_faces(
            id_image=saved_paths['id_image'],
            passport_image=saved_paths['passport_image'],
            selfie_image=saved_paths['selfie_image']
        )

        if 'error' in face_result:
            return jsonify({
                'success': False,
                'error': f"Face verification failed: {face_result['error']}"
            }), 500

        # Step 2: Enhanced Liveness Check on Video
        lv = LiveVideo()
        video_path = saved_paths['video']
        liveness_result = lv.detect_head_movement(
            video_path, 
            saved_paths['selfie_image']
        )
        
        # Construct final result
        return jsonify({
            'success': True,
            'same_person': face_result['same_person'],  # New field
            'all_documents_match': face_result['same_person'],  # Alias for backward compatibility
            'liveness_details': {
                'head_movement_detected': liveness_result['head_movement_detected'],
                'head_rotation_detected': liveness_result['head_rotation_detected'],
                'lips_moving': liveness_result['lips_moving'],  # Updated key
                'match_selfie_video': liveness_result['face_match']
            },
            'face_verification_details': {
                'match_id_selfie': face_result['match_id_selfie'],
                'match_passport_selfie': face_result['match_passport_selfie'],
                'match_id_passport': face_result['match_id_passport'],
                'same_person': face_result['same_person'],  # Also include here
                'details': face_result['details']
            }
        })

    except Exception as e:
        logger.error(f"Verification error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': RESPONSE_MESSAGES['FACE_MATCH_FAILED'].format(str(e))
        }), 500

    finally:
        # Clean up uploaded files
        cleanup_temp_files(list(saved_paths.values()))

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

@app.route('/extract_id_name', methods=['POST'])
def extract_id_name():
    """
    Endpoint to extract name from ID image using OCR
    Expected payload: {'id_image': image_file}
    Returns JSON with extracted name
    """
    if 'id_image' not in request.files:
        return jsonify({
            'success': False,
            'error': RESPONSE_MESSAGES['MISSING_FILE'].format('id_image')
        }), 400

    file = request.files['id_image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': 'Invalid file'
        }), 400

    try:
        # Save uploaded file
        filename = f"{uuid.uuid4().hex}.{file.filename.rsplit('.', 1)[1].lower()}"
        file_path = save_uploaded_file(file, filename)

        # Extract name using OCR
        ocr_id = OcrId()
        extracted_name = ocr_id.extract_name_from_image(file_path)

        if not extracted_name:
            return jsonify({
                'success': False,
                'error': 'Name not found in the image'
            }), 400

        return jsonify({
            'success': True,
            'name': extracted_name
        })

    except Exception as e:
        logger.error(f"ID OCR extraction error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': RESPONSE_MESSAGES['EXTRACTION_ERROR'].format(str(e))
        }), 500

    finally:
        # Clean up uploaded file
        if 'file_path' in locals():
            cleanup_temp_files([file_path])

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)