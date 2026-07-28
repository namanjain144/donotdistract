# do not disturb

AI-powered focus tracking that runs entirely in your browser. No downloads, no cloud processing, no data leaving your device.

## What it does

Do Not Disturb watches you through your webcam during a work or study session and tells you how focused you were afterwards. It detects three types of unfocused behaviour:

- **Distraction** — you looked away from the screen (checked your phone, turned to talk to someone, stared out the window)
- **Fatigue** — your eyes closed for too long or you yawned
- **Absence** — you left your desk entirely

At the end of your session, you get a focus score and a detailed breakdown showing exactly when and why you lost focus.

## How it works

The app uses [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) to detect 478 points on your face from your webcam feed. From those points, it calculates:

**EAR (Eye Aspect Ratio)** — measures how open your eyes are by computing the ratio of vertical to horizontal eye distances. When EAR drops below your personal threshold, your eyes are closed.

**MAR (Mouth Aspect Ratio)** — same formula applied to mouth landmarks. Detects yawning as an early sign of fatigue.

**Head pose (pitch and yaw)** — measures which direction your head is pointing by tracking where your nose sits relative to your eyes. Detects when you turn away from the screen.

**Blink rate and movement** — tracked as informational indicators shown on the dashboard but not included in the focus score, as [research shows](https://doi.org/10.3390/s26030889) these signals alone are not reliable enough for scoring.

All processing happens locally in your browser using WebAssembly. Your webcam feed is never sent anywhere.

## Privacy

Your camera feed never leaves your device. The MediaPipe model runs entirely in your browser via WebAssembly. No video frames, no facial data and no images are ever transmitted to any server. Only session summaries (focus score, event counts, timestamps) are stored, and in the current version even those stay in your browser's localStorage.

## Getting started

Visit the app and follow the three-step flow:

1. **Calibrate** (20 seconds, first time only) — the app learns what your face looks like when you're focused by guiding you through a series of poses
2. **Start a session** — choose how long you want to focus (15, 25, 45 or 60 minutes)
3. **Review your results** — see your focus score, a colour-coded timeline and a breakdown of what caused your unfocused time

## Focus score

The focus score represents the percentage of your session that you were focused:

```
total unfocused = distracted time + fatigued time
grace period    = 5% of session duration (normal blinking and glancing)
adjusted        = max(0, total unfocused - grace period)
focus score     = (session duration - adjusted) / session duration × 100
```

Absent time (when you leave your desk) is excluded from the calculation — it doesn't count for or against you.

The 5% grace period means normal human behaviour (blinking, briefly glancing at your coffee, shifting in your chair) doesn't penalise your score.

## Calibration

Fixed thresholds don't work across different faces. Someone with naturally smaller eyes would constantly be flagged as drowsy. The calibration step captures your personal baselines:

- **Baseline** — what your EAR, MAR and head pose look like when you're sitting normally
- **Extremes** — what your EAR looks like with eyes closed, MAR during a yawn, head pose when turned away
- **Thresholds** — automatically calculated as the midpoint between your baseline and your extremes

Calibration data is saved locally and persists across sessions. You only need to recalibrate if you change your desk setup or camera position.

## Research

Based on the paper:

> Zambrano, T. et al. (2026). *Driver Monitoring System Using Computer Vision for Real-Time Detection of Fatigue, Distraction and Emotion via Facial Landmarks and Deep Learning.* Sensors, 26(3), 889. [doi:10.3390/s26030889](https://doi.org/10.3390/s26030889)

The paper's driving safety system was adapted for desk-work focus tracking. The core signals (EAR, MAR, head pose) and their mathematical formulas are directly from the paper. Key adaptations include per-user calibration (the paper used fixed thresholds), session logging instead of real-time alarms, and the addition of blink rate and movement tracking as informational indicators.

Additional research references:

- Blink rate as an attention indicator: [Elevated Blink Rates Predict Mind Wandering](https://doi.org/10.31083/j.jin2301016) (Journal of Integrative Neuroscience, 2025)
- Attention estimation via blink detection: [ALEBk: Feasibility Study of Attention Level Estimation via Blink Detection](https://arxiv.org/abs/2101.01761) (arXiv, 2021)

## Tech stack

- **React** with Vite
- **MediaPipe Face Landmarker** (JavaScript/WebAssembly, runs client-side)
- **Tailwind CSS** for styling
- **Recharts** for dashboard visualisations
- **localStorage** for data persistence

## Local development

```bash
git clone https://github.com/yourusername/donotdisturb.git
cd donotdisturb
npm install
npm run dev
```

Open `http://localhost:5173` in Chrome or Firefox. Grant camera permission when prompted.

## Browser support

- **Chrome** — fully supported (recommended)
- **Firefox** — fully supported
- **Safari** — limited WebRTC support, may have issues with camera access
- **Edge** — fully supported (Chromium-based)

## Project structure

```
src/
├── lib/
│   ├── mediapipe.js       Load and initialise Face Landmarker
│   ├── detection.js       EAR, MAR, head pose, blink and movement calculations
│   ├── scoring.js         Focus score calculation
│   └── storage.js         localStorage helpers
├── hooks/
│   ├── useWebcam.js       Camera access
│   ├── useDetection.js    Detection loop on video frames
│   └── useSession.js      Session timer, data collection, events
├── screens/
│   ├── Welcome.jsx        Landing screen
│   ├── Questionnaire.jsx  Onboarding questions
│   ├── Calibration.jsx    Guided calibration flow
│   ├── SessionSetup.jsx   Duration selection
│   ├── SessionActive.jsx  Live tracking with HUD
│   └── Results.jsx        Post-session dashboard
└── components/
    ├── WebcamView.jsx     Video element with overlay
    ├── HUD.jsx            Session overlay panel
    ├── Timeline.jsx       Colour-coded session timeline
    ├── StatCard.jsx       Stats display card
    └── EventCard.jsx      Event count card
```

## Roadmap

- [ ] Cloud sync for cross-device session history
- [ ] User accounts and authentication
- [ ] Session history trends and weekly reports
- [ ] Suggested break times based on focus patterns
- [ ] Team/classroom mode for shared focus sessions
- [ ] Mobile browser support

## Licence

MIT
