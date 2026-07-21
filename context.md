# Focus tracker project

A webcam-based distraction and fatigue detection tool that monitors your physical attributes and eye movements during a work session, then produces a dashboard showing how focused you were.

## Research foundation

Based on the paper: **"Driver Monitoring System Using Computer Vision for Real-Time Detection of Fatigue, Distraction and Emotion via Facial Landmarks and Deep Learning"** by Zambrano et al. (Sensors, 2026). The paper's driving safety system was adapted for desk-work focus tracking.

Paper link: https://doi.org/10.3390/s26030889

## Core concepts

### How it works

The system uses Google's MediaPipe Face Landmarker, a pretrained neural network that detects 468 landmark points on a face from a standard webcam feed. No specialised hardware is needed — just a USB or built-in webcam at 640x480 resolution.

From those 468 landmarks, the system extracts three signals every frame (approximately 30 times per second). These signals are combined to determine whether the user is focused, fatigued or distracted.

### What is a pretrained model?

MediaPipe Face Landmarker was trained by Google on thousands of face images where humans manually marked landmark positions. The training process adjusted millions of internal weights until the model could accurately predict those positions on any new face. The `.task` file downloaded during setup contains those final, already-adjusted weights. No training is required to use it — you feed it an image and it returns 468 coordinates.

### What is Euclidean distance?

The straight-line distance between two points, calculated using the Pythagorean theorem:

```
distance = sqrt((x1 - x2)² + (y1 - y2)²)
```

This is used throughout the project to measure gaps between facial landmarks (how open the eye is, how open the mouth is).

## Signal 1: EAR (Eye Aspect Ratio) — fatigue detection

From Section 2.4 of the paper.

### What it measures

How open or closed the eyes are. When EAR drops below a threshold, the eyes are closed — indicating drowsiness or fatigue.

### Formula

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 × ||p1 - p4||)
```

Where p1 and p4 are the horizontal corners of the eye, p2/p6 are upper eyelid points, and p3/p5 are lower eyelid points. The numerator measures the vertical opening (averaged across two pairs for stability), and the denominator measures the horizontal width.

### MediaPipe landmark indices

- Right eye: 33 (outer corner), 160 (upper left), 158 (upper right), 133 (inner corner), 153 (lower right), 144 (lower left)
- Left eye: 362, 385, 387, 263, 373, 380

### Threshold

The paper uses a fixed threshold of 0.23 based on statistical analysis of 27 participants (mean EAR of 0.257, standard deviation of 0.021). This value sits approximately 1.3 standard deviations below the mean.

**Important:** Thresholds should be calibrated per user. People with naturally smaller eyes may have a lower baseline EAR, leading to false positives with a fixed threshold.

### Interpretation

- EAR above threshold: eyes open (normal)
- EAR below threshold: eyes closed (fatigue indicator)

## Signal 2: MAR (Mouth Aspect Ratio) — yawn detection

From Section 2.5 of the paper.

### What it measures

How open the mouth is. A sustained high MAR indicates yawning, which is an early sign of fatigue.

### Formula

Identical structure to EAR:

```
MAR = (||p2 - p6|| + ||p3 - p5||) / (2 × ||p1 - p4||)
```

Applied to mouth landmarks instead of eye landmarks.

### MediaPipe landmark indices

- Mouth: 61 (left corner), 65 (upper lip left), 63 (upper lip right), 67 (right corner), 64 (lower lip right), 66 (lower lip left)

### Threshold

The paper does not specify a fixed MAR threshold. This must be calibrated per user by observing the MAR value with mouth closed versus during a yawn and setting the threshold between those values.

**Note:** In our implementation with the current MediaPipe version, MAR decreases when the mouth opens (inverted from the paper's description). The threshold check uses `<` instead of `>`. MAR is also unreliable when the head is turned — landmarks distort at angles. The system only checks MAR when the head is roughly facing forward.

### Calibrated values (personal)

- Mouth closed: approximately 0.31
- Yawning: drops below 0.29
- Threshold set to: 0.29

## Signal 3: Head pose (pitch and yaw) — distraction detection

From Section 2.6 of the paper.

### What it measures

Which direction the head is pointing. Yaw measures left/right rotation (shaking head "no") and pitch measures up/down tilt (nodding "yes"). When either value exceeds a threshold for a sustained period, the user is looking away from the screen.

### How it works

The system uses a geometric approach: it measures where the nose sits relative to the eyes and chin. When the head turns right, the nose shifts right of the midpoint between the eyes. When the head tilts down, the nose drops lower relative to the eyes-to-chin line.

```python
# Yaw: nose offset from eye midpoint, normalised by eye distance
yaw = ((nose.x - face_center_x) / eye_distance) * 90

# Pitch: nose vertical position relative to eyes-to-chin line
nose_ratio = (nose.y - face_center_y) / face_height
pitch = (nose_ratio - 0.37) * 150  # 0.37 calibrated per user
```

### Key landmarks used

- Nose tip: landmark 1
- Chin: landmark 152
- Left eye corner: landmark 33
- Right eye corner: landmark 263

### Thresholds

- Yaw threshold: 25 degrees (looking too far left or right)
- Pitch threshold: 20 degrees (looking too far up or down)

**Note:** The pitch formula constant (0.37) is calibrated per user. This value centres the pitch reading around zero when looking straight at the camera.

### Grace periods

The paper uses temporal constraints — distraction is only flagged when deviation persists beyond a set duration, avoiding false positives from natural glances (checking a second monitor, looking at notes briefly). This is noted for future implementation.

## Tech stack

- **Python 3.10+**
- **OpenCV (cv2)** — webcam access, image processing, video display
- **MediaPipe 0.10.35** — pretrained face landmark detection (new Tasks API)
- **NumPy** — numerical operations for head pose calculation
- **Model file:** `face_landmarker.task` (downloaded from Google's model repository)

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install mediapipe opencv-python numpy

# Download the pretrained model
curl -O https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

# Run
python focus_tracker.py
```

Press 'q' to quit the application.

## Current state

### What works

- Real-time EAR calculation with eyes open/closed detection
- Real-time MAR calculation with yawn detection (only when facing forward)
- Real-time head pose estimation with distraction detection
- All three signals displayed on the webcam feed simultaneously
- Console output of current values

### What is next

1. **Session logging** — write timestamped data (EAR, MAR, pitch, yaw, state) to a CSV or SQLite database every 0.5 seconds during a session
2. **Focus scoring** — after a session ends, compute overall focus percentage, count distraction events and yawns, identify longest focus streak
3. **Post-session dashboard** — visual timeline of focus state, breakdown by distraction type, session-over-session trends
4. **Grace periods** — implement temporal filtering so brief glances away are not flagged as distractions
5. **Calibration step** — a short calibration phase at the start of each session to set personalised thresholds for EAR, MAR and head pose

## Product vision

A desktop application that runs during work sessions, processing all webcam data locally on the user's device. Only session summaries (focus score, timestamps, event counts) are sent to a cloud backend for cross-device access and historical tracking. The webcam feed is never stored or transmitted — privacy is the core differentiator.

### System design overview

```
User's device (private)
├── Webcam → MediaPipe → Signal engine (EAR, MAR, pose)
├── Session logger (SQLite)
└── Focus scorer → Session summary (JSON)
        │
        ▼  HTTPS (summary only, no video)
Cloud backend
├── REST API (FastAPI)
├── Database (PostgreSQL)
└── Web dashboard (React + charts)
```

## Key lessons learned

- **Fixed thresholds do not work universally.** The paper's EAR threshold of 0.23 and their MAR behaviour did not match the current MediaPipe version or individual facial geometry. Every threshold needed per-user calibration.
- **MediaPipe API has changed.** The older `mp.solutions.face_mesh` API is deprecated. The current version uses `mp.tasks.python.vision.FaceLandmarker` with a downloadable `.task` model file.
- **MAR is unreliable at angles.** When the head is turned, mouth landmarks distort, producing false yawn readings. The fix is to only evaluate MAR when the head is facing forward.
- **solvePnP gave unstable results** for head pose estimation in this setup. A simpler geometric approach (measuring nose position relative to eye midpoint) produced more stable and usable pitch/yaw values.
- **Signal checks must be ordered carefully.** Variables must be defined before they are used in condition checks. The MAR check depends on the yaw value from head pose, so head pose must be calculated first.

## References

- Zambrano, T. et al. (2026). Driver Monitoring System Using Computer Vision for Real-Time Detection of Fatigue, Distraction and Emotion via Facial Landmarks and Deep Learning. *Sensors*, 26(3), 889.
- Dewi, C. et al. (2022). Eye Aspect Ratio for Real-Time Drowsiness Detection. *Electronics*, 11, 3183.
- Lugaresi, C. et al. (2019). MediaPipe: A Framework for Building Perception Pipelines. arXiv.
- 3Blue1Brown. Neural Networks series (YouTube). Foundational understanding of neural network concepts used in this project.