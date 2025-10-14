### TNTFrog — NAI (Narzędzia Sztucznej Inteligencji) project

This repository contains a simple console board game with a basic AI opponent, built as a university project for NAI (Narzędzia Sztucznej Inteligencji) classes.

Authors: Mateusz Anikiej and Aleksander Kunkowski

## Tech stack

- **Language**: Python (requires >= 3.13)
- **AI framework**: `easyAI`
- **Package/Environment manager**: `uv`

## Prerequisites

1. Python 3.13 installed (ensure `python --version` shows 3.13+)
2. `uv` installed (https://docs.astral.sh/uv/getting-started/installation/)
   - Linux/macOS (curl): `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

## Setup

Use `uv` to create a virtual environment and install dependencies from `pyproject.toml`.

Option A: using Makefile

```sh
make setup
```

Option B: using uv directly

```sh
uv sync
```

## Run

Start the game in the console.

Option A: using Makefile

```sh
make run
```

Option B: using uv directly

```sh
uv run python src/main.py
```

While playing, type one of the allowed moves: `up`, `down`, `left`, `right`, `upleft`, `upright`, `downleft`, `downright`.

## Game rules

The game is played on the Square board of odd size (minimum 5x5). Player 1 begins first turn. Each player starts in the opposite corner of the board and take alternating turns moving on the board. Each player can move to any of the 8 neighboring cells (orthogonal or diagonal), using the commands listed above. A move is considered legal if it's within board bounds, target square is unoccupied by other player and and no player was positioned on this field in the previous turns. Cells previously occupied by a player become blocked and are marked as `-1`. A player loses if they have no legal moves on their turn; thus, the opponent wins.
