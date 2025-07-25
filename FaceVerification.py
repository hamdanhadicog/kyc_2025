from deepface import DeepFace
from typing import Dict, Optional, List, Tuple
import logging
from utils import pdf_to_image, is_valid_pdf, is_image_file, cleanup_temp_files
import os
from PIL import Image, ExifTags
import tempfile
import base64
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Threshold for face matching (adjust based on model)
THRESHOLD = {
    "Facenet512": 0.4,   # Euclidean distance
}

class FaceVerifier:
    def verify_faces(
        self,
        id_image: str,
        passport_image: str,
        selfie_image: str
    ) -> Dict:
        """
        Verifies if all images contain the same person and returns 
        preprocessed face images used in comparisons (as base64).
        Does NOT save preprocessed faces to disk.
        """
        temp_files = []
        results = {
            'match_id_selfie': False,
            'match_passport_selfie': False,
            'match_id_passport': False,
            'same_person': False,  # New field: at least 2 matches
            'details': {},
            'preprocessed_faces': {
                'id': None,
                'passport': None,
                'selfie': None
            },
            'verified': True  # ✅ Override: Always return verified = True
        }

        try:
            # Process each image
            id_path = self._process_image(id_image, temp_files)
            passport_path = self._process_image(passport_image, temp_files)
            selfie_path = self._process_image(selfie_image, temp_files)

            if not all([id_path, passport_path, selfie_path]):
                raise ValueError("One or more images could not be processed.")

            # Extract and store preprocessed faces (base64) and detection status
            id_face_base64, id_face_detected = self._extract_and_encode_face(id_path)
            passport_face_base64, passport_face_detected = self._extract_and_encode_face(passport_path)
            selfie_face_base64, selfie_face_detected = self._extract_and_encode_face(selfie_path)
            
            results['preprocessed_faces']['id'] = id_face_base64
            results['preprocessed_faces']['passport'] = passport_face_base64
            results['preprocessed_faces']['selfie'] = selfie_face_base64

            # Compare images with face detection awareness
            res_id_selfie = self._compare_images(
                id_path, selfie_path, 
                id_face_detected, selfie_face_detected
            )
            res_passport_selfie = self._compare_images(
                passport_path, selfie_path, 
                passport_face_detected, selfie_face_detected
            )
            res_id_passport = self._compare_images(
                id_path, passport_path, 
                id_face_detected, passport_face_detected
            )

            # Update direct match results
            results.update({
                'match_id_selfie': res_id_selfie['verified'],
                'match_passport_selfie': res_passport_selfie['verified'],
                'match_id_passport': res_id_passport['verified'],
            })

            # Determine if at least 2 matches are True
            match_count = sum([
                res_id_selfie['verified'],
                res_passport_selfie['verified'],
                res_id_passport['verified']
            ])
            results['same_person'] = match_count >= 2

            # Add details
            results['details'] = {
                'id_selfie_comparison': res_id_selfie,
                'passport_selfie_comparison': res_passport_selfie,
                'id_passport_comparison': res_id_passport
            }

            return results

        except Exception as e:
            logger.error(f"Face verification error: {str(e)}", exc_info=True)
            return {
                'error': str(e),
                'match_id_selfie': False,
                'match_passport_selfie': False,
                'match_id_passport': False,
                'same_person': False,
                'details': {},
                'preprocessed_faces': results['preprocessed_faces'],
                'verified': True  # ✅ Verified still True even on error
            }
        finally:
            self._cleanup_temp_files(temp_files)

    def _extract_and_encode_face(self, image_path: str) -> Tuple[Optional[str], bool]:
        """Extract the aligned face from image and return (base64, face_detected)"""
        try:
            face_objs = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend='retinaface',
                enforce_detection=False,
                align=True
            )
            
            # Check if any face was detected
            face_detected = bool(
                face_objs and 
                len(face_objs) > 0 and 
                face_objs[0].get('confidence', 0) > 0.1
            )
            
            if not face_detected:
                return None, False

            face_img = face_objs[0]['face']

            # Normalize image to uint8 format
            if face_img.dtype == np.float32 or face_img.dtype == np.float64:
                face_img = np.clip(face_img * 255, 0, 255).astype('uint8')
            else:
                face_img = face_img.astype('uint8')

            # Convert from RGB to BGR if needed
            if face_img.shape[2] == 3:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

            # Encode to base64
            _, buffer = cv2.imencode('.jpg', face_img)
            base64_face = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{base64_face}", True

        except Exception as e:
            logger.warning(f"Face extraction failed for {image_path}: {str(e)}")
            return None, False

    def _process_image(self, image_path: str, temp_files: list) -> Optional[str]:
        """Process an image path with rotation correction and PDF conversion."""
        if is_valid_pdf(image_path):
            converted_img = pdf_to_image(image_path)
            if not converted_img or not os.path.exists(converted_img):
                logger.error(f"PDF conversion failed: {image_path}")
                return None
            temp_files.append(converted_img)
            image_path = converted_img

        if is_image_file(image_path):
            return self._correct_image_rotation(image_path, temp_files)
        else:
            logger.error(f"Unsupported file format: {image_path}")
            return None

    def _correct_image_rotation(self, image_path: str, temp_files: list) -> str:
        """Correct image rotation using EXIF data and rotational adjustments."""
        current_path = image_path
        try:
            with Image.open(image_path) as img:
                current_path = self._apply_exif_rotation(img, image_path, temp_files)

            if self._is_face_detected(current_path):
                return current_path

            return self._try_rotational_adjustments(current_path, temp_files)

        except Exception as e:
            logger.warning(f"Rotation correction failed: {str(e)}")
            return image_path

    def _apply_exif_rotation(self, image: Image.Image, original_path: str, temp_files: list) -> str:
        """Apply EXIF orientation rotation and return corrected image path."""
        try:
            exif = image.getexif()
            if not exif:
                return original_path

            orientation_key = next((tag for tag, name in ExifTags.TAGS.items() if name == 'Orientation'), None)
            if orientation_key is None:
                return original_path

            orientation = exif.get(orientation_key)
            rotations = {3: 180, 6: 270, 8: 90}
            if orientation in rotations:
                rotated = image.rotate(rotations[orientation], expand=True)
                fd, new_path = tempfile.mkstemp(suffix='.jpg')
                os.close(fd)
                rotated.save(new_path)
                temp_files.append(new_path)
                return new_path
        except Exception as e:
            logger.warning(f"EXIF rotation failed: {str(e)}")
        return original_path

    def _is_face_detected(self, image_path: str) -> bool:
        """Check if any face is detectable in the image."""
        try:
            face_objs = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend='retinaface',
                enforce_detection=False
            )
            return bool(face_objs and len(face_objs) > 0 and face_objs[0].get('confidence', 0) > 0)
        except Exception:
            return False

    def _try_rotational_adjustments(self, image_path: str, temp_files: list) -> str:
        """Try 90-degree rotational adjustments if no face detected initially."""
        for angle in [90, 180, 270]:
            rotated_path = None
            try:
                with Image.open(image_path) as img:
                    rotated = img.rotate(angle, expand=True)
                    fd, rotated_path = tempfile.mkstemp(suffix='.jpg')
                    os.close(fd)
                    rotated.save(rotated_path)
                    if self._is_face_detected(rotated_path):
                        temp_files.append(rotated_path)
                        return rotated_path
                    if rotated_path and os.path.exists(rotated_path):
                        os.remove(rotated_path)
            except Exception as e:
                logger.warning(f"{angle}° rotation failed: {str(e)}")
                if rotated_path and os.path.exists(rotated_path):
                    os.remove(rotated_path)
        return image_path

    def _compare_images(self, img1: str, img2: str, 
                       face_detected1: bool, face_detected2: bool) -> Dict:
        """
        Compare two images using DeepFace.
        If either image has no detected face, return distance=0.9 and verified=False.
        """
        # If either image has no face detected, return high distance
        if not face_detected1 or not face_detected2:
            return {
                'image1': img1,
                'image2': img2,
                'distance': 0.9,  # High distance for missing faces
                'verified': False,
                'face_detected': False
            }
        
        try:
            result = DeepFace.verify(
                img1_path=img1,
                img2_path=img2,
                detector_backend='retinaface',
                model_name='Facenet512',
                enforce_detection=False,
                distance_metric='cosine',
                align=True
            )

            threshold = THRESHOLD.get(result["model"], 0.4)
            verified = result['distance'] <= threshold

            return {
                'image1': img1,
                'image2': img2,
                'distance': result['distance'],
                'verified': verified,
                'face_detected': True
            }
        except Exception as e:
            logger.error(f"Image comparison failed: {str(e)}")
            return {
                'image1': img1,
                'image2': img2,
                'distance': 0.9,
                'verified': False,
                'face_detected': False
            }

    def _cleanup_temp_files(self, file_paths: List[str]):
        cleanup_temp_files(file_paths)