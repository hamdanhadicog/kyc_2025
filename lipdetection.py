"""
🎤 Enhanced Speaking Detection with MediaPipe Face Mesh (Silent Mode)
Uses 468 high-precision landmarks for superior accuracy.
Returns True if any speaking is detected, False otherwise.
"""

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# =============================================================================
# PARAMETERS
# =============================================================================
LIP_SEPARATION_THRESHOLD = 0.03  # Tune based on testing
MIN_FRAMES_SPEAKING = 2
SMOOTHING_FACTOR = 0.9

def rotate_frame(frame, angle):
    """
    Rotate video frame by specified angle.
    """
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return frame

def is_upright_face(face_landmarks, frame_width, frame_height):
    """
    Determines if the face is upright by checking:
      - Nose lies between the two eyes (horizontally)
      - Nose lies below both eyes (vertically)
    Returns True if face is upright, False otherwise.
    """
    def get_pixel_coords(landmark_idx):
        landmark = face_landmarks.landmark[landmark_idx]
        x = landmark.x * frame_width
        y = landmark.y * frame_height
        return np.array([x, y])

    try:
        left_eye = get_pixel_coords(33)   # Left eye inner corner
        right_eye = get_pixel_coords(263)  # Right eye inner corner
        nose_tip = get_pixel_coords(1)     # Nose tip
    except IndexError:
        return False

    # Validate coordinates are within bounds
    for pt in [left_eye, right_eye, nose_tip]:
        if not (0 <= pt[0] < frame_width and 0 <= pt[1] < frame_height):
            return False

    # Horizontal check: nose should lie between eyes
    leftmost_eye_x = min(left_eye[0], right_eye[0])
    rightmost_eye_x = max(left_eye[0], right_eye[0])
    nose_x_ok = leftmost_eye_x <= nose_tip[0] <= rightmost_eye_x

    # Vertical check: nose should be below both eyes
    eye_center_y = (left_eye[1] + right_eye[1]) / 2
    nose_y_ok = nose_tip[1] > eye_center_y

    return nose_x_ok and nose_y_ok

def determine_rotation_angle(cap):
    """
    Determines optimal rotation angle (0, 90, 180, 270) to make face upright.
    Tests first 5 frames with each rotation angle.
    """
    candidate_angles = [0, 90, 180, 270]
    num_test_frames = 5
    best_angle = 0
    best_score = 0

    temp_face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    for angle in candidate_angles:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        upright_count = 0
        frames_processed = 0
        for _ in range(num_test_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = rotate_frame(frame, angle)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = temp_face_mesh.process(rgb_frame)
            frames_processed += 1
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                frame_height, frame_width = frame.shape[:2]
                if is_upright_face(face_landmarks, frame_width, frame_height):
                    upright_count += 1
        if upright_count > best_score:
            best_score = upright_count
            best_angle = angle
        if best_score == num_test_frames:
            break

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return best_angle

def is_head_rotated(face_landmarks, frame_width, frame_height, threshold=0.5):
    """
    Detects if the head is significantly rotated (yaw) based on nose position.
    Returns True if head is rotated, False if facing forward.
    """
    def get_pixel_coords(landmark_idx):
        landmark = face_landmarks.landmark[landmark_idx]
        x = landmark.x * frame_width
        y = landmark.y * frame_height
        return np.array([x, y])

    try:
        left_eye = get_pixel_coords(33)
        right_eye = get_pixel_coords(263)
        nose_tip = get_pixel_coords(1)
    except IndexError:
        return False

    eye_center = (left_eye + right_eye) / 2
    nose_offset_x = nose_tip[0] - eye_center[0]
    eye_distance = np.linalg.norm(left_eye - right_eye)
    if eye_distance == 0:
        return False

    normalized_offset = abs(nose_offset_x / eye_distance)
    return normalized_offset > threshold

def calculate_relative_lip_distance(landmarks, frame_width, frame_height, is_head_rotated_flag):
    """
    Calculate lip distance using different normalization depending on head rotation.
    If head is rotated: use inter-eye distance.
    """
    if not landmarks.multi_face_landmarks:
        return 0.0

    face_landmarks = landmarks.multi_face_landmarks[0]

    def get_pixel_coords(landmark_idx):
        landmark = face_landmarks.landmark[landmark_idx]
        x = landmark.x * frame_width
        y = landmark.y * frame_height
        return np.array([x, y])

    try:
        upper_point = get_pixel_coords(13)  # Upper lip center
        lower_point = get_pixel_coords(14)  # Lower lip center
    except IndexError:
        return 0.0

    lip_distance = np.linalg.norm(upper_point - lower_point)

    if not is_head_rotated_flag:
        try:
            left_eye = get_pixel_coords(33)
            right_eye = get_pixel_coords(263)
        except IndexError:
            return 0.0

        eye_distance = np.linalg.norm(left_eye - right_eye)
        if eye_distance == 0:
            return 0.0
        return lip_distance / eye_distance
    else:
        return 0.0

def detect_speaking_in_video(video_path):
    """
    Main function to detect speaking in a video.
    Returns True if any speaking is detected, False otherwise.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return False

    # Auto-detect correct rotation
    rotation_angle = determine_rotation_angle(cap)

    # Initialize variables
    prev_smoothed_separation = None
    speaking_frame_count = 0
    found_speaking = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = rotate_frame(frame, rotation_angle)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        frame_height, frame_width = frame.shape[:2]

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            current_head_rotated = is_head_rotated(face_landmarks, frame_width, frame_height)

            if current_head_rotated:
                lip_separation = 0.0
            else:
                lip_separation = calculate_relative_lip_distance(results, frame_width, frame_height, False)

            # Apply smoothing
            if prev_smoothed_separation is None:
                smoothed_separation = lip_separation
            else:
                smoothed_separation = (SMOOTHING_FACTOR * prev_smoothed_separation) + (
                    (1 - SMOOTHING_FACTOR) * lip_separation
                )
            prev_smoothed_separation = smoothed_separation

            # Speaking detection logic
            if smoothed_separation > LIP_SEPARATION_THRESHOLD:
                speaking_frame_count += 1
                if speaking_frame_count >= MIN_FRAMES_SPEAKING:
                    found_speaking = True
                    break  # No need to continue once speaking is confirmed
            else:
                speaking_frame_count = 0

    cap.release()
    return found_speaking

