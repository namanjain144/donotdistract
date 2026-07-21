"""One-command entry point: calibrate (if needed), then run a focus session.

Usage:
    python run.py
"""

import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_PATH = os.path.join(BASE_DIR, "calibration.json")
CALIBRATE_SCRIPT = os.path.join(BASE_DIR, "calibrate.py")
TRACKER_SCRIPT = os.path.join(BASE_DIR, "focus_tracker.py")


def should_calibrate():
    """Decide whether to run calibration, asking the user if one already exists."""
    if not os.path.exists(CALIBRATION_PATH):
        print("No calibration found - starting calibration.")
        return True

    answer = input("Calibration already exists. Recalibrate? [y/N] ").strip().lower()
    return answer == "y"


if should_calibrate():
    result = subprocess.run([sys.executable, CALIBRATE_SCRIPT])
    if result.returncode != 0:
        print("Calibration did not complete - not starting a session.")
        sys.exit(result.returncode)

subprocess.run([sys.executable, TRACKER_SCRIPT])
