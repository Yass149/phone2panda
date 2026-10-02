.PHONY: setup validate-pilots validate-dataset phase4a test lint check

UV_ENV = UV_CACHE_DIR=$(CURDIR)/.cache/uv UV_PYTHON_INSTALL_DIR=$(CURDIR)/.local/share/uv/python
UV = $(UV_ENV) ./tools/run-uv.sh

setup:
	./tools/bootstrap.sh

validate-pilots:
	.venv/bin/python scripts/validate_pilots.py --config configs/pilot_validation.yaml

validate-dataset:
	.venv/bin/python scripts/validate_dataset.py --config configs/dataset_validation.yaml

phase4a:
	MUJOCO_GL=cgl .venv/bin/python scripts/run_phase4a.py --config configs/phase4a.yaml

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check .

check: lint test validate-pilots
