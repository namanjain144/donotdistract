import cv2
import csv
import json
import math
import os
import subprocess
import sys
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np


# Set up the Face Landmarker with the new Tasks API
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
)
detector = vision.FaceLandmarker.create_from_options(options)

# Landmark indices from the paper (Section 2.4)
# Right eye: p1=33, p2=160, p3=158, p4=133, p5=153, p6=144
# Left eye:  p1=362, p2=385, p3=387, p4=263, p5=373, p6=380
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 65, 63, 67, 64, 66]
FACE_3D_POINTS = np.array([
    [0.0, 0.0, 0.0],        # Nose tip
    [0.0, -63.6, -12.5],    # Chin
    [-43.3, 32.7, -26.0],   # Left eye corner
    [43.3, 32.7, -26.0],   # Right eye corner
    [-28.9, -28.9, -24.1],  # Left mouth corner
    [28.9, -28.9, -24.1],   # Right mouth corner
], dtype=np.float64)
EAR_THRESHOLD = 0.23 # From the paper (Section 3.4)
MAR_THRESHOLD = 0.29
# Which MediaPipe landmarks correspond to those 6 points
POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]

PITCH_THRESHOLD = 20
YAW_THRESHOLD = 25   # degrees - looking left/right too far

# Session logging settings
SAMPLE_INTERVAL_SECONDS = 0.5
SESSIONS_DIR = "sessions"


def distance(p1, p2):
    """Euclidean distance between two landmark points."""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def calculate_ear(landmarks, eye_indices):
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    p1, p4 = horizontal corners of the eye
    p2, p6 = upper eyelid points
    p3, p5 = lower eyelid points
    """
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]

    vertical_1 = distance(p2, p6)  # Upper to lower eyelid (pair 1)
    vertical_2 = distance(p3, p5)  # Upper to lower eyelid (pair 2)
    horizontal = distance(p1, p4)  # Corner to corner

    if horizontal == 0:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def calculate_mar(landmarks,m_indices):
    """
    MAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    p1, p4 = horizontal corners of the eye
    p2, p6 = upper eyelid points
    p3, p5 = lower eyelid points
    """
    p1 = landmarks[m_indices[0]]
    p2 = landmarks[m_indices[1]]
    p3 = landmarks[m_indices[2]]
    p4 = landmarks[m_indices[3]]
    p5 = landmarks[m_indices[4]]
    p6 = landmarks[m_indices[5]]

    vertical_1 = distance(p2, p6)  # Upper to lower eyelid (pair 1)
    vertical_2 = distance(p3, p5)  # Upper to lower eyelid (pair 2)
    horizontal = distance(p1, p4)  # Corner to corner

    if horizontal == 0:
        return 0.0

    mar = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return mar

def calculate_head_pose(landmarks, frame_width, frame_height):
    nose = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    chin = landmarks[152]

    # Yaw: how far is nose from the midpoint between the eyes (horizontal)
    face_center_x = (left_eye.x + right_eye.x) / 2
    eye_distance = abs(right_eye.x - left_eye.x)

    if eye_distance == 0:
        return 0.0, 0.0

    yaw = ((nose.x - face_center_x) / eye_distance) * 90

    # Pitch: how far nose is vertically relative to eyes and chin
    face_center_y = (left_eye.y + right_eye.y) / 2
    face_height = chin.y - face_center_y

    if face_height == 0:
        return 0.0, 0.0

    nose_ratio = (nose.y - face_center_y) / face_height
    pitch = (nose_ratio - 0.37) * 150

    return pitch, yaw


def ask_session_duration_minutes():
    """Ask the user how long they'd like to focus for, in minutes."""
    while True:
        raw_value = input("How many minutes would you like to focus for? ")
        try:
            minutes = float(raw_value)
            if minutes > 0:
                return minutes
            print("Please enter a number greater than zero.")
        except ValueError:
            print("Please enter a valid number.")


def determine_state(face_detected, ear, mar, pitch, yaw):
    """Work out the current focus state from the latest signals."""
    if not face_detected:
        return "absent"

    facing_forward = abs(yaw) <= YAW_THRESHOLD and abs(pitch) <= PITCH_THRESHOLD

    if not facing_forward:
        return "distracted"

    if ear < EAR_THRESHOLD or mar < MAR_THRESHOLD:
        return "fatigued"

    return "focused"


# Ask the user for a session length before opening the webcam
session_duration_minutes = ask_session_duration_minutes()
session_duration_seconds = session_duration_minutes * 60

# Prepare the sessions directory and file names, timestamped to the second
os.makedirs(SESSIONS_DIR, exist_ok=True)
session_start_wall_time = time.localtime()
session_timestamp_label = time.strftime("%Y-%m-%d_%H-%M-%S", session_start_wall_time)
csv_path = os.path.join(SESSIONS_DIR, f"session_{session_timestamp_label}.csv")
json_path = os.path.join(SESSIONS_DIR, f"session_{session_timestamp_label}.json")

csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp", "ear", "mar", "pitch", "yaw", "state"])

# Event tracking, kept in memory for the end-of-session summary
look_away_events = []   # list of (start_time, end_time)
eye_closure_events = [] # list of (start_time, end_time)
yawn_events = []        # list of timestamps

# In-progress event trackers (None when not currently in that event)
look_away_start = None
eye_closure_start = None
was_yawning = False  # tracks whether the previous forward-facing sample was mid-yawn

# Per-state cumulative time, built up one sample at a time
state_totals_seconds = {
    "focused": 0.0,
    "distracted": 0.0,
    "fatigued": 0.0,
    "absent": 0.0,
}

session_start_time = time.time()
last_sample_time = session_start_time
next_sample_due = session_start_time

# Open webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit early")
print("-" * 40)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    elapsed_seconds = now - session_start_time
    remaining_seconds = max(0.0, session_duration_seconds - elapsed_seconds)

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to MediaPipe Image format (needs RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Run Face Landmarker
    result = detector.detect(mp_image)

    face_detected = bool(result.face_landmarks)
    avg_ear = 0.0
    mouth_mar = 0.0
    pitch = 0.0
    yaw = 0.0

    if face_detected:
        landmarks = result.face_landmarks[0]

        # Calculate EAR for both eyes
        right_ear = calculate_ear(landmarks, RIGHT_EYE)
        left_ear = calculate_ear(landmarks, LEFT_EYE)
        mouth_mar = calculate_mar(landmarks,MOUTH)
        avg_ear = (right_ear + left_ear) / 2.0

        # Determine state
        if avg_ear < EAR_THRESHOLD:
            state = "EYES CLOSED"
            colour = (0, 0, 255)  # Red
        else:
            state = "EYES OPEN"
            colour = (0, 255, 0)  # Green

        # Calculate head pose
        h, w, _ = frame.shape
        pitch, yaw = calculate_head_pose(landmarks, w, h)

        # MAR state (only check when facing forward)
        if abs(yaw) < YAW_THRESHOLD and mouth_mar < MAR_THRESHOLD:
            mouth_state = "YAWN"
            mouth_colour = (0, 0, 255)
        else:
            mouth_state = ""
            mouth_colour = (0, 255, 0)

        # Head pose state
        if abs(yaw) > YAW_THRESHOLD:
            head_state = "LOOKING AWAY"
            head_colour = (0, 0, 255)
        elif abs(pitch) > PITCH_THRESHOLD:
            head_state = "LOOKING UP/DOWN"
            head_colour = (0, 0, 255)
        else:
            head_state = ""
            head_colour = (0, 255, 0)

        # Display head pose
        cv2.putText(frame, f"Pitch: {pitch:.1f} Yaw: {yaw:.1f}", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, head_colour, 2)
        cv2.putText(frame, head_state, (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, head_colour, 2)

        # Display on the video feed
        cv2.putText(frame, f"EAR: {avg_ear:.3f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
        cv2.putText(frame, state, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)

        cv2.putText(frame, f"MAR: {mouth_mar:.3f}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, mouth_colour, 2)

        cv2.putText(frame, mouth_state, (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, mouth_colour, 2)
        # Also print to console

        print(f"EAR: {avg_ear:.3f} | {state}", end="\r")
        print(f"MAR: {mouth_mar:.3f} | {mouth_state}", end="\r")
    else:
        cv2.putText(frame, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Work out the overall focus state for this frame
    current_state = determine_state(face_detected, avg_ear, mouth_mar, pitch, yaw)
    facing_forward = abs(yaw) <= YAW_THRESHOLD and abs(pitch) <= PITCH_THRESHOLD

    # Track look-away events (start/end pairs) based on the distracted state
    if current_state == "distracted":
        if look_away_start is None:
            look_away_start = now
    else:
        if look_away_start is not None:
            look_away_events.append((look_away_start, now))
            look_away_start = None

    # Track sustained eye closure events, only meaningful while facing forward
    if face_detected and facing_forward and avg_ear < EAR_THRESHOLD:
        if eye_closure_start is None:
            eye_closure_start = now
    else:
        if eye_closure_start is not None:
            eye_closure_events.append((eye_closure_start, now))
            eye_closure_start = None

    # Track yawns as single events, logged on the moment the yawn starts
    is_yawning_now = face_detected and facing_forward and mouth_mar < MAR_THRESHOLD
    if is_yawning_now and not was_yawning:
        yawn_events.append(now)
    was_yawning = is_yawning_now

    # Log a CSV row every SAMPLE_INTERVAL_SECONDS
    if now >= next_sample_due:
        csv_writer.writerow([
            f"{now:.3f}",
            f"{avg_ear:.4f}",
            f"{mouth_mar:.4f}",
            f"{pitch:.2f}",
            f"{yaw:.2f}",
            current_state,
        ])
        state_totals_seconds[current_state] += now - last_sample_time
        last_sample_time = now
        next_sample_due += SAMPLE_INTERVAL_SECONDS

    # Countdown timer overlay
    minutes_left = int(remaining_seconds // 60)
    seconds_left = int(remaining_seconds % 60)
    cv2.putText(frame, f"Time left: {minutes_left:02d}:{seconds_left:02d}", (20, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Focus Tracker - EAR", frame)

    stop_requested = (cv2.waitKey(1) & 0xFF == ord("q"))
    time_is_up = elapsed_seconds >= session_duration_seconds

    if stop_requested or time_is_up:
        # Close out any events still in progress when the session ends
        if look_away_start is not None:
            look_away_events.append((look_away_start, now))
        if eye_closure_start is not None:
            eye_closure_events.append((eye_closure_start, now))
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
csv_file.close()

# --- Build the end-of-session summary ---
actual_duration_seconds = time.time() - session_start_time

total_focused_seconds = state_totals_seconds["focused"]
total_distracted_seconds = state_totals_seconds["distracted"]
total_fatigued_seconds = state_totals_seconds["fatigued"]
total_absent_seconds = state_totals_seconds["absent"]

look_away_total_seconds = sum(end - start for start, end in look_away_events)

total_unfocused_time = total_distracted_seconds + total_fatigued_seconds
bias = session_duration_seconds * 0.05  # 5% grace for normal blinking and glancing
adjusted_unfocused = max(0.0, total_unfocused_time - bias)

if actual_duration_seconds > 0:
    focus_score = ((actual_duration_seconds - adjusted_unfocused) / actual_duration_seconds) * 100
else:
    focus_score = 0.0

summary = {
    "session_duration_minutes": session_duration_minutes,
    "actual_duration_seconds": actual_duration_seconds,
    "total_focused_seconds": total_focused_seconds,
    "total_distracted_seconds": total_distracted_seconds,
    "total_fatigued_seconds": total_fatigued_seconds,
    "total_absent_seconds": total_absent_seconds,
    "look_away_count": len(look_away_events),
    "look_away_total_seconds": look_away_total_seconds,
    "eye_closure_count": len(eye_closure_events),
    "yawn_count": len(yawn_events),
    "focus_score": focus_score,
}

with open(json_path, "w") as json_file:
    json.dump(summary, json_file, indent=2)

print("\nSession complete.")
print(f"Focus score: {focus_score:.1f}%")
print(f"CSV saved to {csv_path}")
print(f"Summary saved to {json_path}")

# Launch the dashboard so the user can review the session in their browser
dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.py")
if os.path.exists(dashboard_path):
    subprocess.Popen([sys.executable, dashboard_path, session_timestamp_label])
