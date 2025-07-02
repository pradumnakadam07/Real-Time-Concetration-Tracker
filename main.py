import cv2
import mediapipe as mp
import numpy as np
import time
from math import hypot

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# Eye landmarks for EAR
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Webcam
cap = cv2.VideoCapture(0)
prev_time = 0
start_time = time.time()

# For average score calculation
concentration_scores = []

def draw_concentration_bar(frame, score):
    bar_x, bar_y = 50, 50
    bar_width, bar_height = 300, 30
    fill_width = int((score / 100) * bar_width)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), 2)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (0, 255, 0), -1)
    cv2.putText(frame, f"Concentration: {int(score)}%", (bar_x, bar_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

def calculate_EAR(landmarks, eye_indices):
    p = [np.array([landmarks[i].x, landmarks[i].y]) for i in eye_indices]
    vertical = hypot(*(p[1] - p[5])) + hypot(*(p[2] - p[4]))
    horizontal = 2 * hypot(*(p[0] - p[3]))
    return vertical / horizontal if horizontal != 0 else 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    curr_time = time.time()
    elapsed_time = int(curr_time - start_time)
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    score = 0

    if result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        # 1. Face detected
        score += 20

        # 2. Eye openness using EAR
        left_ear = calculate_EAR(landmarks, LEFT_EYE)
        right_ear = calculate_EAR(landmarks, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2
        ear_scaled = np.clip((avg_ear - 0.15) / (0.30 - 0.15), 0, 1)
        score += ear_scaled * 40

        # 3. Face centered (nose x near 0.5)
        nose_x = landmarks[1].x
        center_dist = abs(nose_x - 0.5)
        center_score = max(0, 1 - (center_dist / 0.1))  # within ±0.1 range
        score += center_score * 40

        # Draw face mesh (light for speed)
        mp_drawing.draw_landmarks(
            frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec
        )

    # Save score for final average
    concentration_scores.append(score)

    # Draw visuals
    draw_concentration_bar(frame, score)
    cv2.putText(frame, f'FPS: {int(fps)}', (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f'Time: {elapsed_time} sec', (w - 180, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Real-Time Concentration Tracker", frame)
    key = cv2.waitKey(1)
    if key & 0xFF == 27:  # ESC to quit
        break

# Final average score
if concentration_scores:
    average_score = sum(concentration_scores) / len(concentration_scores)
else:
    average_score = 0

# Display result screen
result_frame = np.zeros((300, 600, 3), dtype=np.uint8)
cv2.putText(result_frame, "Session Ended", (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
cv2.putText(result_frame, f"Average Concentration: {int(average_score)}%", (120, 180),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
cv2.imshow("Session Summary", result_frame)
cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
