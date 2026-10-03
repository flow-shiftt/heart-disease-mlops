PY ?= python
IMAGE ?= cardiorisk-api:1.0.0

.PHONY: install data eda train test lint format serve notebooks mlflow-ui docker-build docker-run compose k8s-deploy k8s-monitoring k8s-smoke clean

install:            ## install runtime + dev dependencies
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install -e . --no-deps

data:               ## download + clean the dataset
	$(PY) scripts/download_data.py

eda: data           ## regenerate EDA figures
	$(PY) scripts/run_eda.py

train: data         ## tune, compare, log to MLflow and package the best model
	$(PY) -m cardiorisk.train

test:
	$(PY) -m pytest -v --cov=cardiorisk --cov-report=term-missing

lint:
	ruff check .

format:
	ruff format . && ruff check --fix .

serve:              ## run the API locally on :8000
	PYTHONPATH=src uvicorn cardiorisk.api.app:app --reload --port 8000

notebooks:
	$(PY) scripts/build_notebooks.py && jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 --name cardiorisk-api $(IMAGE)

compose:            ## API + Prometheus + Grafana
	docker compose up --build

k8s-deploy:         ## minikube: build image inside cluster and deploy
	minikube image load $(IMAGE)
	kubectl apply -k deploy/k8s
	kubectl -n cardiorisk rollout status deploy/cardiorisk-api

k8s-monitoring:
	kubectl apply -k deploy/monitoring
	kubectl -n cardiorisk rollout status deploy/prometheus deploy/grafana

k8s-smoke:
	bash scripts/smoke_test.sh http://$$(kubectl -n cardiorisk get svc cardiorisk-api -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ reports/*.xml
