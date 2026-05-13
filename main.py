import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Configuration
model_path = 'hand_landmarker.task' 

# Hardcoded connections so we never have to call mp.solutions again
# This represents every line in the hand skeleton (Wrist to fingers, etc.)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),    # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)          # Palm/Knuckles
]

# 2. Setup the Landmarker Options
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

# 3. Initialize the Detector and Webcam
with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    print("Skeleton Hand Tracker Active! Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        frame_timestamp_ms = int(time.time() * 1000)
        detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # 4. Drawing Logic
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                
                # Draw the lines using our hardcoded list
                for connection in HAND_CONNECTIONS:
                    start_lm = hand_landmarks[connection[0]]
                    end_lm = hand_landmarks[connection[1]]
                    
                    p1 = (int(start_lm.x * frame.shape[1]), int(start_lm.y * frame.shape[0]))
                    p2 = (int(end_lm.x * frame.shape[1]), int(end_lm.y * frame.shape[0]))
                    
                    cv2.line(frame, p1, p2, (255, 255, 255), 2)

                # Draw the green dots
                for landmark in hand_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        cv2.imshow('Hand Tracker 2026', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()