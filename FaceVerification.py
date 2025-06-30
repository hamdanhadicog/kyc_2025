from deepface import DeepFace
from typing import Dict, Optional, List
import logging
from utils import pdf_to_image, is_valid_pdf, is_image_file
import os

logger = logging.getLogger(__name__)

threshold=0.4

class FaceVerifier:

    def verify_faces(
        self,
        id_image: str,
        passport_image: str,
        selfie_image: str
    ) -> Dict:
        """
        Verifies if all images contain the same person by comparing:
        - Selfie vs ID
        - Selfie vs Passport
        - ID vs Passport

        Returns a dictionary with detailed results.
        """

        temp_files = []
        results = {
            'match_id_selfie': False,
            'match_passport_selfie': False,
            'match_id_passport': False,
            'all_match': False,
            'details': {}
        }

        try:
            # Process each image
            id_path = self._process_image(id_image, temp_files)
            passport_path = self._process_image(passport_image, temp_files)
            selfie_path = self._process_image(selfie_image, temp_files)

            if not all([id_path, passport_path, selfie_path]):
                raise ValueError("One or more images could not be processed.")

            # Compare selfie with ID
            logger.info(f"Comparing ID ({id_path}) and Selfie ({selfie_path})")
            res_id_selfie = self._compare_images(id_path, selfie_path)
            results['match_id_selfie'] = res_id_selfie['verified']

            # Compare selfie with Passport
            logger.info(f"Comparing Passport ({passport_path}) and Selfie ({selfie_path})")
            res_passport_selfie = self._compare_images(passport_path, selfie_path)
            results['match_passport_selfie'] = res_passport_selfie['verified']

            # Compare ID with Passport
            logger.info(f"Comparing ID ({id_path}) and Passport ({passport_path})")
            res_id_passport = self._compare_images(id_path, passport_path)
            results['match_id_passport'] = res_id_passport['verified']

            # Check if all comparisons say "verified"
            results['all_match'] = (
                res_id_selfie['verified'] and 
                res_passport_selfie['verified'] and 
                res_id_passport['verified']
            )

            # Add detailed comparison info
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
                'all_match': False,
                'details': {}
            }
        finally:
            self._cleanup_temp_files(temp_files)

    def _process_image(self, image_path: str, temp_files: list) -> Optional[str]:
        """Process an image path. If it's a PDF, convert it."""
        if is_valid_pdf(image_path):
            converted_img = pdf_to_image(image_path)
            if not converted_img or not os.path.exists(converted_img):
                logger.error(f"PDF conversion failed: {image_path}")
                return None
            temp_files.append(converted_img)
            return converted_img
        elif is_image_file(image_path):
            return image_path
        else:
            logger.error(f"Unsupported file format: {image_path}")
            return None

    def _compare_images(self, img1: str, img2: str) -> Dict:
        """Compare two images using DeepFace."""
        result = DeepFace.verify(
            img1_path=img1,
            img2_path=img2,
            detector_backend='retinaface',
            model_name='Facenet512',
            enforce_detection=False,
            
        )

        verified = result['distance'] <= threshold

        return {
            'image1': img1,
            'image2': img2,
            'distance': result['distance'],   
            'verified': verified
        }

    def _cleanup_temp_files(self, file_paths: List[str]):
        from utils import cleanup_temp_files
        cleanup_temp_files(file_paths)