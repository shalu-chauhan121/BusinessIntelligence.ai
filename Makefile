# Convenience targets. Everything here is a plain command you can also run by hand.
.PHONY: help setup backend frontend test demo preview validate validate-agent clean

help:
	@echo "make setup          install backend and frontend dependencies"
	@echo "make backend        run the API on :8000"
	@echo "make frontend       run the UI on :5173"
	@echo "make test           run the backend test suite"
	@echo "make demo           run the four-stage pipeline headless and print the result"
	@echo "make preview        rebuild docs/ui-preview.html from the latest demo run"
	@echo "make validate       score root-cause analysis against the planted ground truth"
	@echo "make validate-agent score the agent loop against the question taxonomy (needs a key)"
	@echo "make clean          remove local data, caches and generated output"

setup:
	cd backend && python3 -m pip install -r requirements.txt
	cd backend && cp -n .env.example .env || true
	cd frontend && npm install
	cd frontend && cp -n .env.example .env || true

backend:
	cd backend && python3 -m uvicorn app.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

test:
	cd backend && python3 -m unittest discover -s tests -t .

demo:
	python3 scripts/run_pipeline_demo.py

preview: demo
	python3 scripts/build_ui_preview.py

# Deterministic and keyless -- safe in CI, and asserted check-by-check by
# backend/tests/test_rca_ground_truth.py.
validate:
	python3 scripts/validate_rca.py

# NOT a CI gate: needs a live ANTHROPIC_API_KEY, costs real money, and its
# tool-selection results are not reproducible run to run. `make test` covers the
# harness itself headless (backend/tests/test_tier_eval.py); this produces the
# tool-usage report that the X5 and progressive-disclosure decisions rest on.
validate-agent:
	python3 scripts/validate_agent.py

clean:
	rm -rf backend/data demo_output
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
