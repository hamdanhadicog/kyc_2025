Here's the complete README.md content ready for your GitHub repository:

```markdown
# KYC Verification API
Python-based KYC verification system with face matching, passport MRZ extraction, ocr of Id, and liveness detection using Flask.

## Features
- ✅ Face verification across multiple images/PDFs
- 🛂 Passport MRZ data extraction
- 🎥 Liveness detection from videos
- 📄 ID card OCR (name extraction)

## Prerequisites
- Python 3.8+
- Poppler utils (for PDF processing)
- GPU recommended for better performance

## Installation
### 1. Clone repository
```bash
git clone https://github.com/hamdanhadicog/kyc_2025
cd kyc_2025
```

### 2. Install Poppler
**Windows**:  
Download installer from [poppler.freedesktop.org](https://poppler.freedesktop.org/) and add to PATH

**Linux (Debian/Ubuntu)**:
```bash
sudo apt-get install poppler-utils
```

**MacOS**:
```bash
brew install poppler
```

### 3. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

## Running the API
```bash
python app.py
```
Server will run at `http://localhost:5000`

## API Endpoints
### 1. Face Verification for id,passport, and selfie. And Live Verification
Verify if multiple images/PDFs belong to the same person

```python
#Verify faces pro
# verify_faces_client.py

import requests

# Configuration
BASE_URL = "http://localhost:5000"

# Paths to your local files
file_paths = {
    'id_image': "p12_id.jpeg",
    'passport_image': "p12_passport.png",
    'selfie_image': "p12_selfie.jpeg",
    'video': "p12_video.mp4"
}

# Open files and prepare them for upload
files = []
opened_files = []  # To keep track for closing later

try:
    for field_name, path in file_paths.items():
        try:
            f = open(path, "rb")
            opened_files.append(f)
            files.append((field_name, f))
        except Exception as e:
            print(f"Error opening file {path}: {e}")
            raise

    # Make the POST request
    response = requests.post(
        f"{BASE_URL}/verify_faces",
        files=files
    )

    # Parse and print result
    result = response.json()
    print("Status Code:", response.status_code)
    print("Response:", result)

finally:
    # Close all opened files
    for f in opened_files:
        f.close()
```

**Sample Response:**
```json
{
  "all_documents_match": false,
  "face_verification_details": {
    "details": {
      "id_passport_comparison": {
        "distance": 0.2719492087883397,
        "image1": "5f0e733a5ef24908a61c00c6b04e3437.jpeg",
        "image2": "994180bef5c047dd8af49c7661d5b3ad.png",
        "verified": true
      },
      "id_selfie_comparison": {
        "distance": 0.47056027626642327,
        "image1": "5f0e733a5ef24908a61c00c6b04e3437.jpeg",
        "image2": "01b89e6a4d30499397a112668a1b57b3.jpeg",
        "verified": false
      },
      "passport_selfie_comparison": {
        "distance": 0.49172486685194416,
        "image1": "994180bef5c047dd8af49c7661d5b3ad.png",
        "image2": "01b89e6a4d30499397a112668a1b57b3.jpeg",
        "verified": false
      }
    },
    "match_id_passport": false,
    "match_id_selfie": false,
    "match_passport_selfie": false
  },
  "liveness_details": {
    "is_live": false
  },
  "liveness_verified": false,
  "success": true
}
```

### 2. Passport MRZ Extraction
Extract machine-readable zone data from passports

```python
with open("passport.png", "rb") as f:
    response = requests.post(
        "http://localhost:5000/extract_passport",
        files={"passport_image": f}
    )
print(response.json())
```

**Sample Response:**
```json
{
  "success": true,
  "passport_data": {
    "mrz_type": "TD3",
    "document_type": "P",
    "country": "USA",
    "surname": "SMITH",
    "given_names": "JOHN DAVID",
    "passport_number": "AB1234567",
    "nationality": "USA",
    "birth_date": "850515",
    "birth_date_readable": "1985-05-15",
    "sex": "M",
    "expiry_date": "300822",
    "expiry_date_readable": "2030-08-22",
    "personal_number": "123456789012",
    "validation_checks": {
      "birth_date_format": true,
      "expiry_date_format": true,
      "passport_number_format": true,
      "passport_number_check": true,
      "birth_date_check": true,
      "expiry_date_check": true,
      "personal_number_check": true,
      "composite_check": true,
      "country_code_check": true
    }
  }
}
```

### 3. Liveness Detection
Detect real person presence from video

```python
with open("video.mp4", "rb") as f:
    response = requests.post(
        "http://localhost:5000/check_liveness",
        files={"video": f}
    )
print(response.json())
```

**Sample Response:**
```json
{
  "success": true,
  "is_live": true
}
```

### 4. ID Card OCR (Name Extraction)
Extract name from ID cards

```python
with open("id_card.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:5000/extract_id_name",
        files={"id_image": f}
    )
print(response.json())
```

**Sample Response:**
```json
{
  "success": true,
  "name": "John David Smith"
}
```

## File Requirements
| Endpoint | Supported Formats | Max Size |
|----------|-------------------|----------|
| `/verify_faces` | JPG, PNG, PDF | 50MB |
| `/extract_passport` | JPG, PNG, PDF | 50MB |
| `/check_liveness` | MP4, MOV, AVI | 50MB |
| `/extract_id_name` | JPG, PNG, PDF | 50MB |

## Technical Specifications
- **Face Verification**: Uses DeepFace with VGG-Face model
- **MRZ Extraction**: EasyOCR with custom validation
- **Liveness Detection**: MediaPipe Face Mesh tracking
- **PDF Processing**: pdf2image with Poppler backend
- **Error Handling**: Comprehensive error logging and cleanup

## Troubleshooting
1. **PDF processing errors**:  
   Verify poppler installation with `pdftoppm -v`
   
2. **Missing dependencies**:  
   Reinstall requirements with `pip install -r requirements.txt`

3. **CUDA errors**:  
   Install CPU-only version:
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

4. **Port conflicts**:  
   Change port in `app.py`:
   ```python
   app.run(host='0.0.0.0', port=5001)  # Change port number
   ```

5. **Large file uploads**:  
   Ensure files are under 50MB and properly formatted

## License
This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

```

Key features of this README:
1. Includes all setup instructions in one place
2. Provides copy-paste ready code samples for each endpoint
3. Shows sample responses for all API calls
4. Contains OS-specific installation steps
5. Includes troubleshooting section for common issues
6. Clearly documents file requirements and formats
7. Maintains consistent formatting for readability
8. Includes technical details about the implementation

The client can directly copy this content into a README.md file in their repository root. For Windows users, you might want to add a note about adding poppler to PATH during installation.