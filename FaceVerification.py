from deepface import DeepFace
from typing import List
import logging
from utils import pdf_to_image, is_valid_pdf, is_image_file  # Import utilities
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceVerifier:

    def verify_faces(self, images_list: List[str]) -> bool:
        """
        Verifies that all provided images (including PDFs) contain the same face.
        Returns True if all match, False otherwise.
        """
        if len(images_list) < 2:
            logger.warning("At least two images are required for verification.")
            return False

        try:
            verified_pairs = set()
            n = len(images_list)

            # Preprocess PDFs into images
            processed_images = []
            temp_files = []

            for img_path in images_list:
                if is_valid_pdf(img_path):
                    logger.info(f"Converting PDF {img_path} to image...")
                    converted_img = pdf_to_image(img_path)
                    if not converted_img or not os.path.exists(converted_img):
                        logger.error(f"Failed to convert PDF: {img_path}")
                        return False
                    processed_images.append(converted_img)
                    temp_files.append(converted_img)
                elif is_image_file(img_path):
                    processed_images.append(img_path)
                else:
                    logger.error(f"Unsupported or invalid file format: {img_path}")
                    return False

            # Now compare all unique pairs
            for i in range(n):
                for j in range(i + 1, n):
                    img1 = processed_images[i]
                    img2 = processed_images[j]
                    pair_key = tuple(sorted((img1, img2)))

                    if pair_key in verified_pairs:
                        continue

                    logger.info(f"Comparing {img1} and {img2}")
                    result = DeepFace.verify(img1_path=img1, img2_path=img2)

                    logger.info(f"Result: {result['verified']}")
                    verified_pairs.add(pair_key)

                    if not result["verified"]:
                        self._cleanup_temp_files(temp_files)
                        return False

            self._cleanup_temp_files(temp_files)
            return True

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            return False

    def _cleanup_temp_files(self, file_paths: List[str]):
        """Helper method to clean up temporary files."""
        from utils import cleanup_temp_files  # Avoid circular imports
        cleanup_temp_files(file_paths)