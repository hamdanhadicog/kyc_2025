import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / 'static' / 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# API Configuration
API_CONFIG = {
    'FACE_MATCH_THRESHOLD': 0.5,
    'LIVENESS_MIN_FRAMES_WITH_FACE': 15,
    'LIVENESS_MIN_MOVEMENT': 8,
    'LIVENESS_MIN_SMILE_FRAMES': 4,
    'TRUST_SCORE_THRESHOLD': 0.9
}

# Update ALLOWED_EXTENSIONS
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'mov', 'avi', 'pdf'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  

# Logging configuration
LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'kyc_service.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
}

# Response messages
RESPONSE_MESSAGES = {
    'MISSING_FILE': 'Missing or empty file: {}',
    'SAVE_ERROR': 'Failed to save uploaded files: {}',
    'EXTRACTION_ERROR': 'Failed to extract information: {}',
    'VERIFICATION_SUCCESS': 'Document verification passed',
    'VERIFICATION_FAILED': 'Document verification failed - trust score below threshold',
    'MRZ_EXTRACTION_FAILED': 'Failed to extract MRZ information',
    'VISUAL_EXTRACTION_FAILED': 'Failed to extract visual information from document',
    'LIVENESS_FAILED': 'Liveness check failed',
    'FACE_MATCH_FAILED': 'Face match failed: {}'
} 