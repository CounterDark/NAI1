# Smart Ad Player

## Authors

- Aleksander Kunkowski
- Mateusz Anikiej

## Prerequisites

1. Python 3.13 installed
2. `uv` installed (https://docs.astral.sh/uv/getting-started/installation/)
3. A webcam connected to your computer.

## Setup

Use `uv` to create a virtual environment and install dependencies.

Option A: using Makefile

```bash
make setup
```

Option B: using uv directly

```bash
uv sync
```

## Running the program

You can run the script using the Makefile or directly with uv.

Option A: using Makefile

```bash
make exe6
```

Option B: using uv directly

```bash
uv run python src/main.py
```

Press `Esc` to exit the application.

## Project Description

This project implements a "Smart Ad Player" that pauses the advertisement when the viewer is not looking at the screen. It uses computer vision techniques (Haar Cascades) to detect eyes in the webcam feed.

### Key Features

- **Eye Detection**: Uses OpenCV and Haar Cascade classifiers to detect eyes in real-time from the webcam feed.
- **Attention Logic**:
  - If eyes are detected (specifically more than 1 rect), the system assumes the user is "PATRZY" (Looking).
  - If no eyes are detected, it assumes "NIE PATRZY" (Not Looking).
- **Smart Playback**:
  - The video advertisement plays only when the user is looking.
  - If the user looks away, the video pauses immediately.
  - When paused, a "REKLAMA ZATRZYMANA" (Ad Paused) overlay appears on the video window.
- **Looping**: The advertisement video automatically loops when it reaches the end.

### Technologies Used

- **OpenCV (`cv2`)**: For video capture, image processing, object detection, and window display.
- **Haar Cascades**: Specifically `haarcascade_eye.xml` for efficient eye detection.
- **Python**: The core programming language.
