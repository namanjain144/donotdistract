"""Guided webcam calibration for focus_tracker.py.

Walks the user through a series of short poses (look straight, close eyes,
yawn, turn head left/right, look down) and derives personalised EAR/MAR/
pitch/yaw thresholds from the measurements, saving them to calibration.json.

Run this once before the first session, or again any time to recalibrate:
    python calibrate.py
"""

import cv2
import json
import math
import subprocess
import time
from datetime import datetime
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Set up the Face Landmarker with the same options as focus_tracker.py
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
)
detector = vision.FaceLandmarker.create_from_options(options)

# Landmark indices, identical to focus_tracker.py
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 65, 63, 67, 64, 66]
# Stable head/body reference points (not eyes/mouth, which move independently)
# used to measure frame-to-frame fidgeting/restlessness
MOVEMENT_LANDMARKS = [10, 152, 234, 454, 1]

CALIBRATION_PATH = "calibration.json"


def distance(p1, p2):
    """Euclidean distance between two landmark points."""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def calculate_ear(landmarks, eye_indices):
    """Same EAR formula as focus_tracker.py."""
    p1, p2, p3, p4, p5, p6 = (landmarks[i] for i in eye_indices)
    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)
    horizontal = distance(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def calculate_mar(landmarks, m_indices):
    """Same MAR formula as focus_tracker.py."""
    p1, p2, p3, p4, p5, p6 = (landmarks[i] for i in m_indices)
    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)
    horizontal = distance(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def calculate_head_pose(landmarks, frame_width, frame_height):
    """Same geometric head pose estimate as focus_tracker.py."""
    nose = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    chin = landmarks[152]

    face_center_x = (left_eye.x + right_eye.x) / 2
    eye_distance = abs(right_eye.x - left_eye.x)
    if eye_distance == 0:
        return 0.0, 0.0
    yaw = ((nose.x - face_center_x) / eye_distance) * 90

    face_center_y = (left_eye.y + right_eye.y) / 2
    face_height = chin.y - face_center_y
    if face_height == 0:
        return 0.0, 0.0
    nose_ratio = (nose.y - face_center_y) / face_height
    pitch = (nose_ratio - 0.37) * 150

    return pitch, yaw


def put_centred_text(frame, text, y, font_scale=1.0, colour=(255, 255, 255), thickness=2):
    """Draw large centred text with a black outline so it reads on any background."""
    (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x = (frame.shape[1] - text_width) // 2
    # Black outline first, then the coloured fill on top
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, colour, thickness, cv2.LINE_AA)


def read_frame(cap):
    """Grab and mirror a frame from the webcam."""
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.flip(frame, 1)


def announce(text):
    """Speak a short phrase out loud via macOS 'say', without blocking the video loop."""
    subprocess.Popen(["say", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_until_ready(cap, upcoming_instruction, step_number, total_steps):
    """Show the upcoming step and wait for the user to press SPACE before starting it."""
    while True:
        frame = read_frame(cap)
        if frame is None:
            continue

        put_centred_text(frame, upcoming_instruction, 160, font_scale=1.1, colour=(255, 255, 255))
        put_centred_text(frame, f"Step {step_number} of {total_steps}", 220, font_scale=0.8, colour=(200, 200, 200))
        put_centred_text(frame, "Press SPACE when ready", 320, font_scale=1.0, colour=(0, 255, 255))

        cv2.imshow("Focus Tracker - Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            break
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            detector.close()
            print("\nCalibration cancelled.")
            raise SystemExit(1)


def run_step(cap, instruction, duration_seconds, step_number, total_steps):
    """Show an instruction and countdown while collecting measurements every frame.

    Returns the median of each measured value across all frames where a face
    was detected during the collection window, plus a movement score and a
    blink-rate estimate (both are only meaningful for the baseline step, but
    are computed generically here so run_step stays a single shared helper).
    """
    ear_samples = []
    ear_timestamped = []  # (timestamp, ear) pairs, used for a post-hoc blink estimate
    mar_samples = []
    pitch_samples = []
    yaw_samples = []
    movement_samples = []  # normalised frame-to-frame displacement of 5 stable landmarks

    previous_positions = None  # set to this frame's landmarks only after all calcs are done
    start_time = time.time()
    end_time = start_time + duration_seconds

    while time.time() < end_time:
        frame = read_frame(cap)
        if frame is None:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = detector.detect(mp_image)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            right_ear = calculate_ear(landmarks, RIGHT_EYE)
            left_ear = calculate_ear(landmarks, LEFT_EYE)
            ear_value = (right_ear + left_ear) / 2.0
            ear_samples.append(ear_value)
            ear_timestamped.append((time.time(), ear_value))
            mar_samples.append(calculate_mar(landmarks, MOUTH))

            h, w, _ = frame.shape
            pitch, yaw = calculate_head_pose(landmarks, w, h)
            pitch_samples.append(pitch)
            yaw_samples.append(yaw)

            # Movement: average displacement of the 5 stable landmarks since the
            # previous frame, normalised by face width so distance-from-camera
            # doesn't affect the number.
            current_positions = [landmarks[i] for i in MOVEMENT_LANDMARKS]
            face_width = distance(landmarks[234], landmarks[454])
            if previous_positions is not None and face_width > 0:
                displacement = sum(
                    distance(current_positions[i], previous_positions[i])
                    for i in range(len(current_positions))
                ) / len(current_positions)
                movement_samples.append(displacement / face_width)
            previous_positions = current_positions  # stored after this frame's calculations

        # Instruction and progress indicator
        put_centred_text(frame, instruction, 160, font_scale=1.1, colour=(255, 255, 255))
        put_centred_text(frame, f"Step {step_number} of {total_steps}", 220, font_scale=0.8, colour=(200, 200, 200))

        # Countdown, shown as whole seconds remaining
        seconds_left = max(0, int(end_time - time.time()) + 1)
        put_centred_text(frame, f"{seconds_left}...", 320, font_scale=1.6, colour=(0, 255, 255))

        # Small green "recording" dot in the corner
        cv2.circle(frame, (30, 30), 10, (0, 255, 0), -1)

        cv2.imshow("Focus Tracker - Calibration", frame)
        cv2.waitKey(1)

    # Post-hoc blink estimate: a quick EAR dip (<0.5s) below ~80% of this
    # window's own median. There's no calibrated ear_threshold yet at this
    # point (that's derived later, from this same baseline), so this uses a
    # threshold relative to the window itself instead.
    blink_count = 0
    if len(ear_timestamped) >= 2:
        temp_threshold = np.median(ear_samples) * 0.8
        was_below_threshold = False
        blink_start_time = None
        for timestamp, ear_value in ear_timestamped:
            if ear_value < temp_threshold and not was_below_threshold:
                blink_start_time = timestamp
                was_below_threshold = True
            if ear_value >= temp_threshold and was_below_threshold:
                if timestamp - blink_start_time < 0.5:
                    blink_count += 1
                was_below_threshold = False

    results = {
        "ear": float(np.median(ear_samples)) if ear_samples else 0.0,
        "mar": float(np.median(mar_samples)) if mar_samples else 0.0,
        "pitch": float(np.median(pitch_samples)) if pitch_samples else 0.0,
        "yaw": float(np.median(yaw_samples)) if yaw_samples else 0.0,
        "movement": float(np.median(movement_samples)) if movement_samples else 0.0,
        "blinks_per_minute": (blink_count * 60.0 / duration_seconds) if duration_seconds > 0 else 0.0,
    }
    return results


def show_final_message(cap, seconds):
    """Show the completion message on the video feed before closing."""
    end_time = time.time() + seconds
    while time.time() < end_time:
        frame = read_frame(cap)
        if frame is None:
            continue

        put_centred_text(frame, "Calibration complete!", 200, font_scale=1.3, colour=(0, 255, 0))
        put_centred_text(frame, "You are good to go", 250, font_scale=0.8, colour=(255, 255, 255))

        cv2.imshow("Focus Tracker - Calibration", frame)
        cv2.waitKey(1)


TOTAL_STEPS = 6

# Minimum acceptable separation between a baseline reading and its extreme.
# Below this, the value is indistinguishable from normal frame-to-frame noise.
MIN_GAP_FRACTION = 0.10   # for EAR/MAR, a relative (%) difference
MIN_ANGLE_GAP_DEGREES = 10  # for pitch/yaw, an absolute difference in degrees


def gap_ok_fraction(baseline, value, min_fraction=MIN_GAP_FRACTION):
    """Check a baseline-vs-extreme gap for EAR/MAR, as a fraction of baseline."""
    if baseline == 0:
        return abs(value) >= min_fraction
    return abs(baseline - value) / abs(baseline) >= min_fraction


def gap_ok_degrees(baseline, value, min_degrees=MIN_ANGLE_GAP_DEGREES):
    """Check a baseline-vs-extreme gap for pitch/yaw, in absolute degrees."""
    return abs(value - baseline) >= min_degrees


def wait_for_redo_decision(cap, message, step_number, total_steps):
    """After a step whose reading looks implausible, ask whether to redo it."""
    while True:
        frame = read_frame(cap)
        if frame is None:
            continue

        put_centred_text(frame, "That reading looks off:", 130, font_scale=0.9, colour=(0, 165, 255))
        put_centred_text(frame, message, 175, font_scale=0.65, colour=(0, 165, 255))
        put_centred_text(frame, f"Step {step_number} of {total_steps}", 220, font_scale=0.8, colour=(200, 200, 200))
        put_centred_text(frame, "Press R to redo, or SPACE to keep it", 320, font_scale=0.9, colour=(0, 255, 255))

        cv2.imshow("Focus Tracker - Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            return True
        if key == ord(" "):
            return False
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            detector.close()
            print("\nCalibration cancelled.")
            raise SystemExit(1)


def run_step_until_good(cap, instruction, duration_seconds, step_number, total_steps, check_fn):
    """Run a calibration step, offering to redo it if the reading looks implausible.

    check_fn takes the step's result dict and returns (is_ok, message_if_not_ok).
    """
    while True:
        wait_until_ready(cap, instruction, step_number, total_steps)
        result = run_step(cap, instruction, duration_seconds, step_number, total_steps)
        announce("Completed")

        is_ok, message = check_fn(result)
        if is_ok:
            return result

        print(f"\nStep {step_number}: {message}")
        announce("That looks off. You can redo this step.")
        if wait_for_redo_decision(cap, message, step_number, total_steps):
            continue  # go again
        return result  # user chose to keep it anyway


cap = cv2.VideoCapture(0)
print("Starting calibration. Follow the on-screen instructions.")
print("-" * 40)

# Step 1: baseline, looking straight ahead - just needs a face to have been detected.
# Runs for 10 seconds (longer than the other steps) so the blink-rate baseline
# has enough time to produce a meaningful estimate.
step1 = run_step_until_good(
    cap, "Look straight at the camera", 10, 1, TOTAL_STEPS,
    check_fn=lambda r: (
        r["ear"] != 0.0 and r["mar"] != 0.0,
        "No face was detected - make sure you're facing the camera.",
    ),
)
baseline_ear = step1["ear"]
baseline_mar = step1["mar"]
baseline_pitch = step1["pitch"]
baseline_yaw = step1["yaw"]
baseline_movement = step1["movement"]
baseline_blinks_per_minute = step1["blinks_per_minute"]

# Step 2: eyes closed - EAR should drop well below baseline
step2 = run_step_until_good(
    cap, "Now close your eyes", 3, 2, TOTAL_STEPS,
    check_fn=lambda r: (
        gap_ok_fraction(baseline_ear, r["ear"]),
        "Your eyes-closed reading was too close to baseline - close your eyes fully.",
    ),
)
closed_ear = step2["ear"]

# Step 3: yawn - MAR should move well away from baseline
step3 = run_step_until_good(
    cap, "Now open your mouth wide like a yawn", 3, 3, TOTAL_STEPS,
    check_fn=lambda r: (
        gap_ok_fraction(baseline_mar, r["mar"]),
        "Your yawn reading was too close to baseline - open your mouth wider.",
    ),
)
yawn_mar = step3["mar"]
mar_direction = "decreases" if yawn_mar < baseline_mar else "increases"

# Step 4: head turned left
step4 = run_step_until_good(
    cap, "Turn your head to the LEFT", 3, 4, TOTAL_STEPS,
    check_fn=lambda r: (
        gap_ok_degrees(baseline_yaw, r["yaw"]),
        "Your left turn was too close to baseline - turn your head further left.",
    ),
)
left_yaw = step4["yaw"]

# Step 5: head turned right
step5 = run_step_until_good(
    cap, "Turn your head to the RIGHT", 3, 5, TOTAL_STEPS,
    check_fn=lambda r: (
        gap_ok_degrees(baseline_yaw, r["yaw"]),
        "Your right turn was too close to baseline - turn your head further right.",
    ),
)
right_yaw = step5["yaw"]

# Step 6: head tilted down
step6 = run_step_until_good(
    cap, "Look DOWN at your lap", 3, 6, TOTAL_STEPS,
    check_fn=lambda r: (
        gap_ok_degrees(baseline_pitch, r["pitch"]),
        "Your downward tilt was too close to baseline - tilt your head down further.",
    ),
)
down_pitch = step6["pitch"]

# --- Threshold calculation ---
ear_threshold = (baseline_ear + closed_ear) / 2
mar_threshold = (baseline_mar + yawn_mar) / 2

# Head pose: 60% of the way from baseline to the extreme gives a comfortable
# margin so normal small movements don't trigger a "looking away" state.
yaw_threshold_left = baseline_yaw + 0.6 * (left_yaw - baseline_yaw)
yaw_threshold_right = baseline_yaw + 0.6 * (right_yaw - baseline_yaw)
yaw_threshold = min(abs(yaw_threshold_left - baseline_yaw), abs(yaw_threshold_right - baseline_yaw))

pitch_threshold = abs(0.6 * (down_pitch - baseline_pitch))

# Soft-signal thresholds: how many times baseline counts as "elevated".
# These are less definitive than head pose/EAR, so the multipliers are gentler.
MOVEMENT_THRESHOLD_MULTIPLIER = 2.0
BLINK_RATE_THRESHOLD_MULTIPLIER = 1.5

announce("Calibration complete")
show_final_message(cap, 3)

cap.release()
cv2.destroyAllWindows()
detector.close()

calibration = {
    "baseline_ear": baseline_ear,
    "closed_ear": closed_ear,
    "ear_threshold": ear_threshold,
    "baseline_mar": baseline_mar,
    "yawn_mar": yawn_mar,
    "mar_threshold": mar_threshold,
    "mar_direction": mar_direction,
    "baseline_pitch": baseline_pitch,
    "baseline_yaw": baseline_yaw,
    "down_pitch": down_pitch,
    "left_yaw": left_yaw,
    "right_yaw": right_yaw,
    "pitch_threshold": pitch_threshold,
    "yaw_threshold": yaw_threshold,
    "baseline_movement": baseline_movement,
    "movement_threshold_multiplier": MOVEMENT_THRESHOLD_MULTIPLIER,
    "baseline_blinks_per_minute": baseline_blinks_per_minute,
    "blink_rate_threshold_multiplier": BLINK_RATE_THRESHOLD_MULTIPLIER,
    "calibrated_at": datetime.now().isoformat(timespec="seconds"),
}

with open(CALIBRATION_PATH, "w") as calibration_file:
    json.dump(calibration, calibration_file, indent=2)

print("\nCalibration complete. Values saved to calibration.json:")
print("-" * 40)
for key, value in calibration.items():
    print(f"{key:18}: {value}")
print("-" * 40)
print("You can now run focus_tracker.py")
