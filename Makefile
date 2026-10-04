.PHONY: setup validate-pilots validate-dataset phase4a phase4b-preflight phase4b phase4c-audit phase4c-preflight phase4c phase4d-preflight phase4d phase4e-diagnostics phase4e-preflight phase4e phase5a phase5b phase6 test lint check readme-results build-demo sanitize-media smoke-demo simulate-demo verify-data evaluate-existing submission-check reproduce-public privacy-history

UV_ENV = UV_CACHE_DIR=$(CURDIR)/.cache/uv UV_PYTHON_INSTALL_DIR=$(CURDIR)/.local/share/uv/python
UV = $(UV_ENV) ./tools/run-uv.sh

setup:
	./tools/bootstrap.sh

validate-pilots:
	.venv/bin/python scripts/validate_pilots.py --config configs/pilot_validation.yaml

validate-dataset:
	.venv/bin/python scripts/validate_dataset.py --config configs/dataset_validation.yaml

phase4a:
	.venv/bin/python scripts/run_phase4a.py --config configs/phase4a.yaml

phase4b-preflight:
	.venv/bin/python scripts/run_phase4b.py --config configs/phase4b.yaml --preflight-only

phase4b:
	.venv/bin/python scripts/run_phase4b.py --config configs/phase4b.yaml --evaluate

phase4c-audit:
	.venv/bin/python scripts/run_phase4c.py --config configs/phase4c.yaml --audit-original

phase4c-preflight:
	.venv/bin/python scripts/run_phase4c.py --config configs/phase4c.yaml --preflight-only

phase4c:
	.venv/bin/python scripts/run_phase4c.py --config configs/phase4c.yaml --evaluate

phase4d-preflight:
	.venv/bin/python scripts/run_phase4d.py --config configs/phase4d.yaml --preflight-only

phase4d:
	.venv/bin/python scripts/run_phase4d.py --config configs/phase4d.yaml --evaluate

phase4e-diagnostics:
	.venv/bin/python scripts/run_phase4e.py --config configs/phase4e.yaml --diagnostics

phase4e-preflight:
	.venv/bin/python scripts/run_phase4e.py --config configs/phase4e.yaml --preflight-only

phase4e:
	.venv/bin/python scripts/run_phase4e.py --config configs/phase4e.yaml --evaluate

phase5a:
	.venv/bin/python scripts/run_phase5a.py --config configs/phase5a.yaml

phase5b:
	.venv/bin/python scripts/run_phase5b.py --config configs/phase5b.yaml

phase6:
	.venv/bin/python scripts/run_phase6.py --config configs/phase6.yaml

readme-results:
	.venv/bin/python scripts/generate_readme_results.py

build-demo:
	./scripts/build_demo.sh

sanitize-media:
	.venv/bin/python scripts/sanitize_public_media.py

smoke-demo:
	.venv/bin/python scripts/check_submission.py --demo-only

simulate-demo:
	.venv/bin/python scripts/run_public_demo.py

verify-data:
	.venv/bin/python scripts/check_public_trajectories.py

evaluate-existing:
	.venv/bin/python scripts/check_submission.py

submission-check:
	.venv/bin/python scripts/check_public_trajectories.py
	.venv/bin/python scripts/generate_readme_results.py --check
	.venv/bin/python scripts/check_submission.py
	.venv/bin/python scripts/check_repository_privacy.py --tracked

privacy-history:
	.venv/bin/python scripts/check_repository_privacy.py --history

reproduce-public: lint test submission-check

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check .

check: lint test validate-pilots
