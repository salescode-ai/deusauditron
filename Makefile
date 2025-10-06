PYTHON ?= python3
PIP ?= pip3

.PHONY: venv install run run-redis test lint format generate

venv:
	$(PYTHON) -m venv venv
	. venv/bin/activate && $(PIP) install --upgrade pip

install:
	. venv/bin/activate && $(PIP) install -r ./requirements.txt

install-dev:
	. venv/bin/activate && $(PIP) install -r ./requirements-dev.txt

run:
	. venv/bin/activate && PYTHONPATH=src uvicorn deusauditron.app:app --host 0.0.0.0 --port 8081

run-redis:
	USE_REDIS=true REDIS_URL=redis://127.0.0.1:6379/0 make run

test:
	. venv/bin/activate && PYTHONPATH=src pytest -q

lint:
	. venv/bin/activate && ruff check .

format:
	. venv/bin/activate && ruff format .

