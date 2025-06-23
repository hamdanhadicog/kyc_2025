from deepface import DeepFace
import logging
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import pdf_to_image, is_valid_pdf, is_image_file
import os
import numpy as np

logger = logging.getLogger(__name__)

class FaceVerifier:

    def verify_faces(self, images_list: List[str]) -> bool:
        if len(images_list) < 2:
            logger.warning("At least two images are required for verification.")
            return False

        try:
            # Convert PDFs to images and validate input files
            processed_images = []
            temp_files = []

            for img_path in images_list:
                if is_valid_pdf(img_path):
                    converted_img = pdf_to_image(img_path)
                    if not converted_img or not os.path.exists(converted_img):
                        logger.error(f"Failed to convert PDF: {img_path}")
                        return False
                    processed_images.append(converted_img)
                    temp_files.append(converted_img)
                elif is_image_file(img_path):
                    processed_images.append(img_path)
                else:
                    logger.error(f"Unsupported file: {img_path}")
                    return False
                
            # Generate all unique pairs
            n = len(processed_images)
            pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    pair = (processed_images[i], processed_images[j])
                    pairs.append(pair)

            # Run comparisons in parallel
            threshold = 70  # Adjust based on model used
            model_name = 'Facenet512'
            results = self._run_parallel_verification(pairs, model_name, threshold)

            # Check if any pair failed
            if not all(results):
                self._cleanup_temp_files(temp_files)
                return False

            self._cleanup_temp_files(temp_files)
            return True

        except Exception as e:
            logger.error(f"Error during verification: {e}", exc_info=True)
            return False

    def _run_parallel_verification(self, pairs: List[Tuple[str, str]], model_name: str, threshold: float) -> List[bool]:
        """Runs face verification for all image pairs in parallel."""
        
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(self._compare_pair, img1, img2, model_name, threshold)
                for img1, img2 in pairs
            ]

            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if not result:
                    # Optional: cancel remaining tasks early
                    pass  # Not shown here for simplicity

            return results

    @staticmethod
    def _compare_pair(img1: str, img2: str, model_name: str, threshold: float) -> bool:
        
        """Compare a single pair of images."""
        logger = logging.getLogger(__name__)
        logger.info(f"Comparing {img1} and {img2}")

        try:
            
            emb1 = DeepFace.represent(img_path=img1, model_name=model_name, enforce_detection=False)
            emb2 = DeepFace.represent(img_path=img2, model_name=model_name, enforce_detection=False)
            dist = np.linalg.norm(np.array(emb1[0]["embedding"]) - np.array(emb2[0]["embedding"]))
            logger.info(f"Distance between {img1} and {img2}: {dist:.2f}")
            return dist <= threshold

        except Exception as e:
            logger.error(f"Error comparing {img1} and {img2}: {e}")
            return False

    def _cleanup_temp_files(self, file_paths: List[str]):
        from utils import cleanup_temp_files
        cleanup_temp_files(file_paths)