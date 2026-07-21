"""FastAPI dashboard that shows results for the most recent (or a given) focus session."""

import csv
import glob
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser

import uvicorn

DASHBOARD_PORT = 8000
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app = FastAPI()
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def find_session_label():
    """Work out which session to display: a CLI arg, or the most recent one on disk."""
    if len(sys.argv) > 1:
        return sys.argv[1]

    csv_files = sorted(glob.glob(os.path.join(SESSIONS_DIR, "session_*.csv")))
    if not csv_files:
        return None
    latest_csv = csv_files[-1]
    filename = os.path.basename(latest_csv)
    return filename[len("session_"):-len(".csv")]


def load_session(session_label):
    """Load the CSV samples and JSON summary for a given session label."""
    csv_path = os.path.join(SESSIONS_DIR, f"session_{session_label}.csv")
    json_path = os.path.join(SESSIONS_DIR, f"session_{session_label}.json")

    if not os.path.exists(csv_path) or not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail=f"Session '{session_label}' not found")

    with open(json_path) as json_file:
        summary = json.load(json_file)

    samples = []
    with open(csv_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            samples.append(row["state"])

    return summary, samples


@app.get("/", response_class=HTMLResponse)
def show_dashboard(request: Request, session: str | None = None):
    session_label = session or find_session_label()
    if session_label is None:
        raise HTTPException(status_code=404, detail="No sessions found in the sessions/ directory")

    summary, samples = load_session(session_label)

    focus_score = summary["focus_score"]
    if focus_score >= 80:
        score_colour = "#2ecc71"  # green
    elif focus_score >= 60:
        score_colour = "#f1c40f"  # amber
    else:
        score_colour = "#e74c3c"  # red

    state_colours = {
        "focused": "#2ecc71",
        "distracted": "#e74c3c",
        "fatigued": "#f1c40f",
        "absent": "#6b7280",
    }

    unfocused_total = summary["total_distracted_seconds"] + summary["total_fatigued_seconds"]
    if unfocused_total > 0:
        distracted_pct = (summary["total_distracted_seconds"] / unfocused_total) * 100
    else:
        distracted_pct = 0.0

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "session_label": session_label,
            "summary": summary,
            "samples": samples,
            "focus_score": round(focus_score, 1),
            "score_colour": score_colour,
            "state_colours": state_colours,
            "distracted_pct": distracted_pct,
        },
    )


def open_browser_when_ready(url, delay_seconds=1.0):
    time.sleep(delay_seconds)
    webbrowser.open(url)


def port_in_use(port):
    """Check whether something is already listening on the given local port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("localhost", port)) == 0


def find_pids_using_port(port):
    """Find the PIDs of any process listening on the given port (macOS/Linux only)."""
    try:
        output = subprocess.check_output(["lsof", "-ti", f"tcp:{port}"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return [int(pid) for pid in output.split()]


def free_up_port(port):
    """If the port is occupied, ask the user whether to kill whatever is holding it."""
    if not port_in_use(port):
        return

    pids = find_pids_using_port(port)
    if not pids:
        print(f"Port {port} is in use but the process holding it could not be identified.")
        return

    answer = input(f"Port {port} is already in use (PID {', '.join(map(str, pids))}). Kill it and continue? [y/N] ")
    if answer.strip().lower() != "y":
        print("Leaving the existing process running. The dashboard cannot start on this port.")
        sys.exit(1)

    for pid in pids:
        os.kill(pid, signal.SIGKILL)

    # Give the OS a moment to actually release the port before uvicorn binds to it
    for _ in range(20):
        if not port_in_use(port):
            break
        time.sleep(0.1)


if __name__ == "__main__":
    free_up_port(DASHBOARD_PORT)
    url = f"http://localhost:{DASHBOARD_PORT}"
    threading.Thread(target=open_browser_when_ready, args=(url,), daemon=True).start()
    uvicorn.run(app, host="localhost", port=DASHBOARD_PORT)
