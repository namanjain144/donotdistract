import cv2
import csv
import json
import math
import os
import time
from collections import deque
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
# Stable head/body reference points (not eyes/mouth, which move independently)
# used to measure frame-to-frame fidgeting/restlessness
MOVEMENT_LANDMARKS = [10, 152, 234, 454, 1]
FACE_3D_POINTS = np.array([
    [0.0, 0.0, 0.0],        # Nose tip
    [0.0, -63.6, -12.5],    # Chin
    [-43.3, 32.7, -26.0],   # Left eye corner
    [43.3, 32.7, -26.0],   # Right eye corner
    [-28.9, -28.9, -24.1],  # Left mouth corner
    [28.9, -28.9, -24.1],   # Right mouth corner
], dtype=np.float64)
# Which MediaPipe landmarks correspond to those 6 points
POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]

# Session logging settings
SAMPLE_INTERVAL_SECONDS = 0.5
SESSIONS_DIR = "sessions"
CALIBRATION_PATH = "calibration.json"


def load_calibration():
    """Load personalised thresholds, or tell the user to calibrate first."""
    if not os.path.exists(CALIBRATION_PATH):
        print("No calibration found. Run python calibrate.py first.")
        raise SystemExit(1)
    with open(CALIBRATION_PATH) as calibration_file:
        return json.load(calibration_file)


def is_yawning(mar, threshold, direction):
    """MAR moves in different directions depending on the camera/face, per calibration."""
    if direction == "decreases":
        return mar < threshold
    return mar > threshold


calibration = load_calibration()
EAR_THRESHOLD = calibration["ear_threshold"]
MAR_THRESHOLD = calibration["mar_threshold"]
MAR_DIRECTION = calibration["mar_direction"]
PITCH_THRESHOLD = calibration["pitch_threshold"]
YAW_THRESHOLD = calibration["yaw_threshold"]
BASELINE_PITCH = calibration["baseline_pitch"]
BASELINE_YAW = calibration["baseline_yaw"]

# Soft "zoned out" signals - added after the original calibration format, so
# an older calibration.json won't have them. Missing/zero baselines disable
# the corresponding check rather than firing on every frame.
BASELINE_MOVEMENT = calibration.get("baseline_movement", 0.0)
MOVEMENT_THRESHOLD_MULTIPLIER = calibration.get("movement_threshold_multiplier", 2.0)
BASELINE_BLINKS_PER_MINUTE = calibration.get("baseline_blinks_per_minute", 0.0)
BLINK_RATE_THRESHOLD_MULTIPLIER = calibration.get("blink_rate_threshold_multiplier", 1.5)

if BASELINE_MOVEMENT <= 0 or BASELINE_BLINKS_PER_MINUTE <= 0:
    print("Note: this calibration predates zoned-out detection (movement/blink baselines missing).")
    print("Recalibrate with python calibrate.py to enable it. Continuing without it for now.")


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


def determine_state(face_detected, ear, mar, pitch, yaw, is_restless, is_blink_rate_elevated):
    """Work out the current focus state from the latest signals.

    is_restless and is_blink_rate_elevated are the debounced/sustained versions
    of the movement and blink-rate checks (see the main loop), not raw
    single-sample threshold crossings - both are soft, noise-prone signals.
    """
    if not face_detected:
        return "absent"

    facing_forward = abs(yaw) <= YAW_THRESHOLD and abs(pitch) <= PITCH_THRESHOLD

    if not facing_forward:
        return "distracted"

    if ear < EAR_THRESHOLD or is_yawning(mar, MAR_THRESHOLD, MAR_DIRECTION):
        return "fatigued"

    if is_restless or is_blink_rate_elevated:
        return "zoned_out"

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
csv_writer.writerow(["timestamp", "ear", "mar", "pitch", "yaw", "movement", "blinks_per_minute", "state"])

# Event tracking, kept in memory for the end-of-session summary.
# All three are (start_time, end_time) pairs, later filtered by MIN_EVENT_DURATION_SECONDS
# so a single noisy frame doesn't get counted as a real event.
look_away_events = []
eye_closure_events = []
yawn_events = []

MIN_EVENT_DURATION_SECONDS = 0.2

# In-progress event trackers (None when not currently in that event)
look_away_start = None
eye_closure_start = None
yawn_start = None

# Per-state cumulative time, built up one sample at a time
state_totals_seconds = {
    "focused": 0.0,
    "distracted": 0.0,
    "fatigued": 0.0,
    "zoned_out": 0.0,
    "absent": 0.0,
}

# --- Zoned-out signal tracking (movement/fidgeting + blink rate) ---
# Rolling window of the last 60 seconds of movement samples, taken once per
# 0.5s tick (60s / 0.5s = 120 samples), gated to frames facing the screen.
movement_window = deque(maxlen=120)
previous_movement_positions = None  # set after all of this frame's calcs are done

blink_timestamps = []       # times of confirmed quick (<0.5s) EAR dips
was_below_ear_threshold = False
blink_start_time = None

# Debounce timers: a soft signal only counts once it's been elevated for a
# sustained period, so a brief noisy spike doesn't flip the logged state.
MOVEMENT_SUSTAINED_SECONDS = 10
BLINK_RATE_SUSTAINED_SECONDS = 30
movement_elevated_since = None
blink_rate_elevated_since = None
is_restless_sustained = False
is_blink_rate_sustained = False

# Running values, refreshed at each 0.5s sample tick, read every frame
rolling_avg_movement = 0.0

# Running totals for the end-of-session averages
movement_sample_sum = 0.0
blinks_per_minute_sample_sum = 0.0
sample_count = 0

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
    raw_movement = 0.0
    blinks_per_minute = 0

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

        # Calculate head pose, offset so "looking straight" reads as ~0,0
        # regardless of this user's camera angle (from calibration)
        h, w, _ = frame.shape
        pitch, yaw = calculate_head_pose(landmarks, w, h)
        pitch -= BASELINE_PITCH
        yaw -= BASELINE_YAW

        # MAR state (only check when facing forward)
        if abs(yaw) < YAW_THRESHOLD and is_yawning(mouth_mar, MAR_THRESHOLD, MAR_DIRECTION):
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

        # Fidget/movement: average displacement of 5 stable landmarks since the
        # previous frame, normalised by face width. Computed every frame for a
        # smooth signal; only fed into the rolling window at the 0.5s sample tick.
        current_movement_positions = [landmarks[i] for i in MOVEMENT_LANDMARKS]
        face_width = distance(landmarks[234], landmarks[454])
        if previous_movement_positions is not None and face_width > 0:
            raw_movement = sum(
                distance(current_movement_positions[i], previous_movement_positions[i])
                for i in range(len(current_movement_positions))
            ) / len(current_movement_positions) / face_width
        else:
            raw_movement = 0.0
        # Stored after all of this frame's calculations, so next frame compares
        # current to previous, never current to current.
        previous_movement_positions = current_movement_positions

        # Blink detection: a quick EAR dip (<0.5s) below threshold is a blink;
        # longer than that is a sustained closure (handled above as "fatigued").
        if avg_ear < EAR_THRESHOLD and not was_below_ear_threshold:
            blink_start_time = now
            was_below_ear_threshold = True
        if avg_ear >= EAR_THRESHOLD and was_below_ear_threshold:
            if now - blink_start_time < 0.5:
                blink_timestamps.append(now)
            was_below_ear_threshold = False
        recent_blinks = [t for t in blink_timestamps if t > now - 60]
        blinks_per_minute = len(recent_blinks)

        # Raw (non-debounced) threshold checks, used only for the on-screen colour
        movement_elevated_raw = BASELINE_MOVEMENT > 0 and rolling_avg_movement > BASELINE_MOVEMENT * MOVEMENT_THRESHOLD_MULTIPLIER
        blink_rate_elevated_raw = BASELINE_BLINKS_PER_MINUTE > 0 and blinks_per_minute > BASELINE_BLINKS_PER_MINUTE * BLINK_RATE_THRESHOLD_MULTIPLIER
        movement_colour = (0, 165, 255) if movement_elevated_raw else (0, 255, 0)
        blink_colour = (0, 165, 255) if blink_rate_elevated_raw else (0, 255, 0)

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

        # Soft zoned-out signals
        cv2.putText(frame, f"Blinks/min: {blinks_per_minute}", (20, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, blink_colour, 2)
        cv2.putText(frame, f"Movement: {rolling_avg_movement:.4f}", (20, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, movement_colour, 2)
        # Also print to console

        print(f"EAR: {avg_ear:.3f} | {state}", end="\r")
        print(f"MAR: {mouth_mar:.3f} | {mouth_state}", end="\r")
    else:
        cv2.putText(frame, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Work out the overall focus state for this frame. Uses whatever the sustained
    # movement/blink-rate flags currently are (refreshed once per 0.5s tick below) -
    # look-away/eye-closure/yawn tracking below don't depend on the zoned_out
    # branch, so a flag being up to 0.5s stale here doesn't affect their accuracy.
    current_state = determine_state(face_detected, avg_ear, mouth_mar, pitch, yaw, is_restless_sustained, is_blink_rate_sustained)
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

    # Track sustained yawns as start/end pairs, same pattern as eye closures.
    # A single noisy frame crossing the threshold isn't a real yawn, so this
    # gets filtered by minimum duration below, alongside the other events.
    is_yawning_now = face_detected and facing_forward and is_yawning(mouth_mar, MAR_THRESHOLD, MAR_DIRECTION)
    if is_yawning_now:
        if yawn_start is None:
            yawn_start = now
    else:
        if yawn_start is not None:
            yawn_events.append((yawn_start, now))
            yawn_start = None

    # Log a CSV row every SAMPLE_INTERVAL_SECONDS
    if now >= next_sample_due:
        # Refresh the movement rolling window (last 60s = 120 samples at 0.5s each),
        # ignoring frames where the head is already turned away per the spec -
        # fidgeting only counts while actually facing the screen.
        if face_detected and facing_forward:
            movement_window.append(raw_movement)
        rolling_avg_movement = (sum(movement_window) / len(movement_window)) if movement_window else 0.0

        # Debounce: a soft signal only flips to "sustained" once it's stayed
        # elevated for a while, so a brief noisy spike doesn't count.
        movement_elevated_now = BASELINE_MOVEMENT > 0 and rolling_avg_movement > BASELINE_MOVEMENT * MOVEMENT_THRESHOLD_MULTIPLIER
        if movement_elevated_now:
            if movement_elevated_since is None:
                movement_elevated_since = now
            is_restless_sustained = (now - movement_elevated_since) >= MOVEMENT_SUSTAINED_SECONDS
        else:
            movement_elevated_since = None
            is_restless_sustained = False

        blink_rate_elevated_now = BASELINE_BLINKS_PER_MINUTE > 0 and blinks_per_minute > BASELINE_BLINKS_PER_MINUTE * BLINK_RATE_THRESHOLD_MULTIPLIER
        if blink_rate_elevated_now:
            if blink_rate_elevated_since is None:
                blink_rate_elevated_since = now
            is_blink_rate_sustained = (now - blink_rate_elevated_since) >= BLINK_RATE_SUSTAINED_SECONDS
        else:
            blink_rate_elevated_since = None
            is_blink_rate_sustained = False

        # Recompute with the freshly updated sustained flags for this tick's log entry
        current_state = determine_state(face_detected, avg_ear, mouth_mar, pitch, yaw, is_restless_sustained, is_blink_rate_sustained)

        csv_writer.writerow([
            f"{now:.3f}",
            f"{avg_ear:.4f}",
            f"{mouth_mar:.4f}",
            f"{pitch:.2f}",
            f"{yaw:.2f}",
            f"{rolling_avg_movement:.5f}",
            blinks_per_minute,
            current_state,
        ])
        state_totals_seconds[current_state] += now - last_sample_time
        last_sample_time = now
        next_sample_due += SAMPLE_INTERVAL_SECONDS

        movement_sample_sum += rolling_avg_movement
        blinks_per_minute_sample_sum += blinks_per_minute
        sample_count += 1

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
        if yawn_start is not None:
            yawn_events.append((yawn_start, now))
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
csv_file.close()

# --- Build the end-of-session summary ---

# Drop events shorter than MIN_EVENT_DURATION_SECONDS - these are landmark
# jitter/noise crossing a threshold for a single frame, not a real event.
# State totals (focused/distracted/fatigued/absent seconds) are unaffected,
# since those reflect actual time spent regardless of event-count noise.
def filter_short_events(events):
    return [(start, end) for start, end in events if end - start >= MIN_EVENT_DURATION_SECONDS]


look_away_events = filter_short_events(look_away_events)
eye_closure_events = filter_short_events(eye_closure_events)
yawn_events = filter_short_events(yawn_events)

actual_duration_seconds = time.time() - session_start_time

total_focused_seconds = state_totals_seconds["focused"]
total_distracted_seconds = state_totals_seconds["distracted"]
total_fatigued_seconds = state_totals_seconds["fatigued"]
total_zoned_out_seconds = state_totals_seconds["zoned_out"]
total_absent_seconds = state_totals_seconds["absent"]

look_away_total_seconds = sum(end - start for start, end in look_away_events)

average_blinks_per_minute = (blinks_per_minute_sample_sum / sample_count) if sample_count > 0 else 0.0
average_movement = (movement_sample_sum / sample_count) if sample_count > 0 else 0.0

# zoned_out is a soft signal (elevated blink rate/fidgeting), not a definitive
# detection like looking away or closed eyes, so it's penalised at half weight.
total_unfocused_time = total_distracted_seconds + total_fatigued_seconds + (total_zoned_out_seconds * 0.5)
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
    "total_zoned_out_seconds": total_zoned_out_seconds,
    "total_absent_seconds": total_absent_seconds,
    "look_away_count": len(look_away_events),
    "look_away_total_seconds": look_away_total_seconds,
    "eye_closure_count": len(eye_closure_events),
    "yawn_count": len(yawn_events),
    "average_blinks_per_minute": average_blinks_per_minute,
    "average_movement": average_movement,
    "focus_score": focus_score,
}

with open(json_path, "w") as json_file:
    json.dump(summary, json_file, indent=2)

def format_minutes_seconds(total_seconds):
    """Render a seconds count as M:SS for readable terminal output."""
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    return f"{minutes}m {seconds:02d}s"


print("\n" + "=" * 40)
print("SESSION COMPLETE")
print("=" * 40)
print(f"Planned duration : {session_duration_minutes:.1f} min")
print(f"Actual duration  : {format_minutes_seconds(actual_duration_seconds)}")
print("-" * 40)
print(f"Focused time     : {format_minutes_seconds(total_focused_seconds)}")
print(f"Distracted time  : {format_minutes_seconds(total_distracted_seconds)}")
print(f"Fatigued time    : {format_minutes_seconds(total_fatigued_seconds)}")
print(f"Zoned-out time   : {format_minutes_seconds(total_zoned_out_seconds)}")
print(f"Absent time      : {format_minutes_seconds(total_absent_seconds)}")
print("-" * 40)
print(f"Look-aways       : {len(look_away_events)} (total {format_minutes_seconds(look_away_total_seconds)})")
print(f"Eye closures     : {len(eye_closure_events)}")
print(f"Yawns            : {len(yawn_events)}")
print(f"Avg blinks/min   : {average_blinks_per_minute:.1f}")
print(f"Avg movement     : {average_movement:.4f}")
print("-" * 40)

# Distraction breakdown: how the unfocused time split across the three causes
unfocused_breakdown_total = total_distracted_seconds + total_fatigued_seconds + total_zoned_out_seconds
if unfocused_breakdown_total > 0:
    print("Unfocused time breakdown:")
    print(f"  Distracted (looking away)     : {total_distracted_seconds / unfocused_breakdown_total * 100:.0f}%")
    print(f"  Fatigued (eyes/yawning)        : {total_fatigued_seconds / unfocused_breakdown_total * 100:.0f}%")
    print(f"  Zoned out (restless/blinking)  : {total_zoned_out_seconds / unfocused_breakdown_total * 100:.0f}%")
    print("-" * 40)

print(f"FOCUS SCORE      : {focus_score:.1f}%")
print("=" * 40)
print(f"CSV saved to {csv_path}")
print(f"Summary saved to {json_path}")
