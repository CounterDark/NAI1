### Cart Pole — NAI (Narzędzia Sztucznej Inteligencji) project

This repository contains a simple simulation of the cart balancing a pole, built as a university project for NAI (Narzędzia Sztucznej Inteligencji) classes.

Authors: Mateusz Anikiej and Aleksander Kunkowski

## Tech stack

- **Language**: Python (requires >= 3.13)
- **Simulation framework**: `gymnasium`
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
make exe2
```

Option B: using uv directly

```sh
uv run python src/main.py
```

### CLI arguments

...

## Game rules

...
