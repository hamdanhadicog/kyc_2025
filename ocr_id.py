import easyocr
from pdf2image import convert_from_path
import os
import uuid

class OcrId:

    def extract_name_from_image(self, file_path):
        """
        Extracts name from an image or PDF file.
        Supports common image formats and PDF.
        """
        # Check if it's a PDF
        if file_path.lower().endswith('.pdf'):
            return self._extract_name_from_pdf(file_path)
        else:
            return self._extract_name_from_image(file_path)

    def _extract_name_from_image(self, image_path):
        """Internal method to extract name from a single image."""
        reader = easyocr.Reader(['en'])  # Adjust language if needed
        result = reader.readtext(image_path)

        name = None
        for detection in result:
            bbox, text, confidence = detection
            if "name" in text.lower():
                try:
                    name = text.split(':', 1)[1].strip()
                except IndexError:
                    name = text.strip()
                break
        return name

    def _extract_name_from_pdf(self, pdf_path):
        """Converts PDF to images and extracts name from each page until found."""
        temp_images = []

        try:
            # Convert PDF to list of PIL images
            images = convert_from_path(pdf_path, dpi=200)  # High DPI for better OCR
            reader = easyocr.Reader(['en'])

            for image in images:
                # Save image temporarily
                temp_image_path = f"{uuid.uuid4()}.jpg"
                image.save(temp_image_path, 'JPEG')
                temp_images.append(temp_image_path)

                # Try extracting name from this image
                name = self._extract_name_from_image(temp_image_path)
                if name:
                    return name

            return None  # If no name found in any page

        finally:
            # Clean up temporary image files
            for img in temp_images:
                if os.path.exists(img):
                    os.remove(img)