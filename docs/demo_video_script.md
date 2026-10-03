# Pipeline demo video — recording script (~5 min)

Record with QuickTime (File → New Screen Recording) or OBS. Speak over each step in your own words.

| # | Time | Show | Say (key points) |
|---|------|------|------------------|
| 1 | 0:00–0:30 | README + `docs/architecture.png` | Problem, three layers: experimentation → CI/CD → serving & monitoring |
| 2 | 0:30–1:15 | `python scripts/download_data.py`, open `notebooks/01_eda.ipynb` | Missing values kept for in-pipeline imputation; strongest signals thal/ca/cp; nominal codes one-hot encoded |
| 3 | 1:15–2:00 | `python -m cardiorisk.train` then `mlflow ui --backend-store-uri sqlite:///mlflow.db` | 3 model families, 5-fold CV, nested runs, artefacts, registry; LR chosen on CV ROC-AUC 0.907 |
| 4 | 2:00–2:30 | `pytest -v` and `ruff check .` | 26 tests: data, features, model, API |
| 5 | 2:30–3:15 | GitHub → Actions → latest run | lint → test → train (+ROC-AUC gate) → docker smoke test → kind deploy; show artefacts and the earlier run that failed at lint |
| 6 | 3:15–3:45 | `docker build …` / `docker run …` / `bash scripts/smoke_test.sh` | Non-root slim image, /predict returns prediction + confidence, 422 on bad input |
| 7 | 3:45–4:30 | `kubectl -n cardiorisk get all,ingress,hpa`, `minikube service cardiorisk-api -n cardiorisk --url`, curl /predict | 2 replicas, probes, LoadBalancer, ingress, HPA |
| 8 | 4:30–5:00 | `python scripts/load_test.py <url>` then Grafana (`kubectl -n cardiorisk port-forward svc/grafana 3000:3000`) | Live latency, error rate, prediction mix, drift histogram; `kubectl logs` JSON logs |

## Bring the environment back up before recording

```bash
colima start && minikube start
kubectl -n cardiorisk get pods            # all Running
minikube service cardiorisk-api -n cardiorisk --url   # keep this terminal open; note the URL
kubectl -n cardiorisk port-forward svc/grafana 3000:3000
```
