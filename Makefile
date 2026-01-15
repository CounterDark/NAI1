SHELL := /bin/sh

.PHONY: setup exe1 exe2 exe3 exe4 exe5 clean

setup:
	uv sync

exe1:
	cd exercise_1 && uv run python src/main.py $(ARGS)

exe2:
	cd exercise_2 && uv run python src/main.py $(ARGS)

exe3:
	cd exercise_3 && uv run python src/main.py $(ARGS)

exe4:
	cd exercise_4 && uv run python src/main.py

exe5:
	cd exercise_5 && uv run python src/main.py $(ARGS)

exe6:
	cd exercise_6 && uv run python src/main.py

exe7:
	cd exercise_7 && uv run python src/main.py $(ARGS)

clean:
	rm -rf .venv
	rm -rf uv.lock
