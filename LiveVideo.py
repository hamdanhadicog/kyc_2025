import cv2
import mediapipe as mp
import numpy as np

class LiveVideo:

    def detect_head_movement(self,video_path):
        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize variables for movement tracking
        initial_nose_x = None
        movement_threshold = 0.03  # Normalized threshold (3% of frame width)
        left_movement = False
        right_movement = False
        frame_count = 0
        max_frames = 300  # Process up to 300 frames (~10 sec at 30fps)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file")
            return False
        
        while cap.isOpened() and frame_count < max_frames:
            success, frame = cap.read()
            if not success:
                break
            
            # Process frame with MediaPipe
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Get nose tip coordinates (landmark index 4)
                nose = face_landmarks.landmark[4]
                height, width, _ = frame.shape
                nose_x = nose.x  # Normalized coordinate (0-1)
                
                # Initialize reference point on first detection
                if initial_nose_x is None:
                    initial_nose_x = nose_x
                    continue
                
                # Check for left movement
                if nose_x < initial_nose_x - movement_threshold:
                    left_movement = True
                    
                # Check for right movement
                if nose_x > initial_nose_x + movement_threshold:
                    right_movement = True
                
                # Early exit if both movements detected
                if left_movement and right_movement:
                    break
            
            frame_count += 1
        
        cap.release()
        return left_movement and right_movement


