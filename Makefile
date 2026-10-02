.PHONY: setup validate-pilots validate-dataset phase4a phase4b-preflight phase4b phase4c-audit phase4c-preflight phase4c phase4d-preflight phase4d phase4e-diagnostics phase4e-preflight phase4e test lint check

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

phase4b-preflight:
	MUJOCO_GL=cgl .venv/bin/python scripts/run_phase4b.py --config configs/phase4b.yaml --preflight-only

phase4b:
	MUJOCO_GL=cgl .venv/bin/python scripts/run_phase4b.py --config configs/phase4b.yaml --evaluate

phase4c-audit:
	MUJOCO_GL=cgl .venv/bin/python scripts/run_phase4c.py --config configs/phase4c.yaml --audit-original

phase4c-preflight:
	MUJOCO_GL=cgl .venv/bin/python scripts/run_phase4c.py --config configs/phase4c.yaml --preflight-only

phase4c:
	MUJOCO_GL=cgl .venv/bin/python scripts/run_phase4c.py --config configs/phase4c.yaml --evaluate

phase4d-preflight:
	MUJOCO_GL=cgl .venv/bin/python scripts/run_phase4d.py --config configs/phase4d.yaml --preflight-only

phase4d:
	MUJOCO_GL=cgl .venv/bin/python scripts/run_phase4d.py --config configs/phase4d.yaml --evaluate

phase4e-diagnostics:
	MUJOCO_GL=cgl .venv/bin/python scripts/run_phase4e.py --config configs/phase4e.yaml --diagnostics

phase4e-preflight:
	MUJOCO_GL=cgl .venv/bin/python scripts/run_phase4e.py --config configs/phase4e.yaml --preflight-only

phase4e:
	MUJOCO_GL=cgl .venv/bin/python scripts/run_phase4e.py --config configs/phase4e.yaml --evaluate

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check .

check: lint test validate-pilots
