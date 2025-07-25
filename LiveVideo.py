import cv2
import mediapipe as mp
import numpy as np
import logging
import os
from lipdetection import detect_speaking_in_video  # Import lip movement detection
from deepface import DeepFace

# Initialize MediaPipe face detector
mp_face_detection = mp.solutions.face_detection

# Settings
SKIP_FRAMES = 3  # Process every Nth frame for speed

class LiveVideo:
    def __init__(self):
        self.face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    
    def detect_head_movement(self, video_path, selfie_image_path):
        # Initialize MediaPipe solutions
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize variables
        initial_nose_x = None
        initial_nose_y = None
        movement_threshold = 0.05
        left_movement = False
        right_movement = False
        up_movement = False
        down_movement = False
        face_match = False
        frame_count = 0
        max_frames = 300
        
        # For head rotation detection
        prev_yaw = None
        head_rotated = False
        rotation_threshold = 15  # degrees
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                'head_movement_detected': False,
                'head_rotation_detected': False,
                'lips_moving': False,  # Changed from smile_detected
                'face_match': False
            }
        
        # Use lip movement detection from det2.py
        lips_moving = detect_speaking_in_video(video_path)
        
        while cap.isOpened() and frame_count < max_frames:
            success, frame = cap.read()
            if not success:
                break
            
            # Process frame with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = face_landmarks.landmark
                img_h, img_w, _ = frame.shape
                
                # Head position detection
                nose = landmarks[4]
                nose_x = nose.x
                nose_y = nose.y
                
                if initial_nose_x is None:
                    initial_nose_x = nose_x
                    initial_nose_y = nose_y
                else:
                    # Detect horizontal movement
                    if nose_x < initial_nose_x - movement_threshold:
                        left_movement = True
                    if nose_x > initial_nose_x + movement_threshold:
                        right_movement = True
                    
                    # Detect vertical movement
                    if nose_y < initial_nose_y - movement_threshold:
                        up_movement = True
                    if nose_y > initial_nose_y + movement_threshold:
                        down_movement = True
                
                # Head rotation detection using 3D pose estimation
                rotation = self.estimate_head_pose(landmarks, img_w, img_h)
                if rotation is not None:
                    yaw, pitch, roll = rotation
                    
                    if prev_yaw is None:
                        prev_yaw = yaw
                    else:
                        # Detect significant rotation change
                        if abs(yaw - prev_yaw) > rotation_threshold:
                            head_rotated = True
                        prev_yaw = yaw
            
            # Face matching every 30 frames to reduce processing
            if frame_count % 30 == 0 and results.multi_face_landmarks:
                try:
                    # Extract face region
                    face_region = self.extract_face_region(frame, landmarks, img_w, img_h)
                    if face_region is not None:
                        # Save temporary face image for comparison
                        temp_face_path = f"temp_face_{frame_count}.jpg"
                        cv2.imwrite(temp_face_path, face_region)
                        
                        # Compare with selfie image
                        result = DeepFace.verify(
                            img1_path=temp_face_path,
                            img2_path=selfie_image_path,
                            model_name='Facenet512',
                            distance_metric='cosine',
                            detector_backend='retinaface',
                            enforce_detection=False
                        )
                        
                        if result['distance'] <= 0.4:  # Same threshold as FaceVerifier
                            face_match = True
                            if os.path.exists(temp_face_path):
                                os.remove(temp_face_path)
                            break
                            
                        # Cleanup temp file
                        if os.path.exists(temp_face_path):
                            os.remove(temp_face_path)
                except Exception as e:
                    logging.error(f"Face matching error: {str(e)}")
            
            frame_count += 1
            
            # Early exit if we've already detected everything we need
            if face_match and head_rotated and (left_movement or right_movement):
                break
        
        cap.release()
        face_mesh.close()
        
        # Determine if any head movement occurred
        head_moved = left_movement or right_movement or up_movement or down_movement
        
        return {
            'head_movement_detected': head_moved,
            'head_rotation_detected': head_rotated,
            'lips_moving': lips_moving,  # Changed from smile_detected
            'face_match': face_match
        }
    
    def estimate_head_pose(self, landmarks, img_w, img_h):
        """Estimate head rotation (yaw, pitch, roll) using facial landmarks"""
        try:
            # 3D model points for reference
            model_points = np.array([
                (0.0, 0.0, 0.0),          # Nose tip
                (0.0, -330.0, -65.0),      # Chin
                (-225.0, 170.0, -135.0),    # Left eye left corner
                (225.0, 170.0, -135.0),     # Right eye right corner
                (-150.0, -150.0, -125.0),   # Left Mouth corner
                (150.0, -150.0, -125.0)     # Right mouth corner
            ])
            
            # 2D image points from landmarks
            image_points = np.array([
                (landmarks[4].x * img_w, landmarks[4].y * img_h),    # Nose tip
                (landmarks[152].x * img_w, landmarks[152].y * img_h), # Chin
                (landmarks[33].x * img_w, landmarks[33].y * img_h),   # Left eye
                (landmarks[263].x * img_w, landmarks[263].y * img_h),  # Right eye
                (landmarks[61].x * img_w, landmarks[61].y * img_h),   # Left mouth
                (landmarks[291].x * img_w, landmarks[291].y * img_h)  # Right mouth
            ], dtype="double")
            
            # Camera internals
            focal_length = img_w
            center = (img_w/2, img_h/2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")
            
            dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
            
            # Solve for pose
            (success, rotation_vector, translation_vector) = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs
            )
            
            # Convert rotation vector to Euler angles
            rotation_mat, _ = cv2.Rodrigues(rotation_vector)
            pose_mat = cv2.hconcat((rotation_mat, translation_vector))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
            
            yaw = euler_angles[1]
            pitch = euler_angles[0]
            roll = euler_angles[2]
            
            return yaw, pitch, roll
        except Exception:
            return None
    
    def extract_face_region(self, frame, landmarks, img_w, img_h):
        """Extract the face region from the frame based on facial landmarks"""
        try:
            # Get bounding box from facial landmarks
            x_min = min(l.x for l in landmarks) * img_w
            y_min = min(l.y for l in landmarks) * img_h
            x_max = max(l.x for l in landmarks) * img_w
            y_max = max(l.y for l in landmarks) * img_h
            
            # Add padding
            padding = 0.2
            width = x_max - x_min
            height = y_max - y_min
            x_min = max(0, int(x_min - padding * width))
            y_min = max(0, int(y_min - padding * height))
            x_max = min(img_w, int(x_max + padding * width))
            y_max = min(img_h, int(y_max + padding * height))
            
            # Extract face region
            face_region = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
            if face_region.size == 0:
                return None
                
            # Resize to consistent size
            return cv2.resize(face_region, (256, 256))
        except Exception:
            return None