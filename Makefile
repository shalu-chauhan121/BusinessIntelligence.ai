# Convenience targets. Everything here is a plain command you can also run by hand.
.PHONY: help setup backend frontend test demo preview clean

help:
	@echo "make setup     install backend and frontend dependencies"
	@echo "make backend   run the API on :8000"
	@echo "make frontend  run the UI on :5173"
	@echo "make test      run the backend test suite"
	@echo "make demo      run the four-stage pipeline headless and print the result"
	@echo "make preview   rebuild docs/ui-preview.html from the latest demo run"
	@echo "make clean     remove local data, caches and generated output"

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

clean:
	rm -rf backend/data demo_output
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
