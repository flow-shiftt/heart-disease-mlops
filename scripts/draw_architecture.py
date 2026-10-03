"""Render docs/architecture.png — the end-to-end system diagram used in the report."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "architecture.png"

LANES = [
    ("1 · Data & experimentation (local / CI)", 7.6, "#fbeeee"),
    ("2 · CI/CD — GitHub Actions", 5.0, "#eef3f8"),
    ("3 · Serving & monitoring — Kubernetes (Minikube / kind)", 2.4, "#eef8f3"),
]

BOXES = {
    # name: (x, y, w, h, label)
    "uci": (0.3, 7.9, 2.0, 1.0, "UCI repository\nprocessed.cleveland.data"),
    "clean": (2.8, 7.9, 2.2, 1.0, "download_data.py\nclean · validate · CSV"),
    "eda": (5.5, 7.9, 1.8, 1.0, "EDA\nnotebook + figures"),
    "train": (7.8, 7.9, 2.6, 1.0, "train.py · GridSearchCV\nLR · RF · GB  (5-fold CV)"),
    "mlflow": (10.9, 7.9, 2.6, 1.0, "MLflow\nruns · metrics · plots\nmodel registry"),
    "pkg": (14.0, 7.9, 2.1, 1.0, "models/\nmodel.joblib\nmetadata.json"),
    "lint": (0.3, 5.3, 1.8, 1.0, "lint\nruff"),
    "test": (2.6, 5.3, 1.8, 1.0, "test\npytest + coverage"),
    "ci_train": (4.9, 5.3, 2.4, 1.0, "train\n+ ROC-AUC ≥ 0.80 gate"),
    "build": (7.8, 5.3, 2.4, 1.0, "docker build\n+ container smoke test"),
    "kind": (10.7, 5.3, 2.6, 1.0, "deploy to kind\nverify /predict + scrape"),
    "arts": (13.8, 5.3, 2.3, 1.0, "artifacts\nreports · model · logs"),
    "client": (0.3, 2.7, 1.8, 1.0, "client\ncurl / app"),
    "ingress": (2.6, 2.7, 2.2, 1.0, "Ingress (nginx)\n/ LoadBalancer Svc"),
    "pods": (5.3, 2.5, 3.0, 1.4, "Deployment · 2–5 pods (HPA)\nFastAPI /predict /health\n/ready /metrics"),
    "prom": (8.9, 2.7, 2.2, 1.0, "Prometheus\npod discovery"),
    "graf": (11.6, 2.7, 2.2, 1.0, "Grafana\ndashboard"),
    "logs": (14.3, 2.7, 1.8, 1.0, "JSON logs\nstdout → kubectl"),
}

ARROWS = [
    ("uci", "clean"), ("clean", "eda"), ("eda", "train"), ("train", "mlflow"), ("mlflow", "pkg"),
    ("lint", "test"), ("test", "ci_train"), ("ci_train", "build"), ("build", "kind"), ("kind", "arts"),
    ("client", "ingress"), ("ingress", "pods"), ("pods", "prom"), ("prom", "graf"),
]


def center(name, side):
    x, y, w, h, _ = BOXES[name]
    return {"r": (x + w, y + h / 2), "l": (x, y + h / 2), "b": (x + w / 2, y), "t": (x + w / 2, y + h)}[side]


def main():
    fig, ax = plt.subplots(figsize=(16.5, 9))
    ax.set_xlim(0, 16.4)
    ax.set_ylim(1.8, 9.6)
    ax.axis("off")
    for title, y, color in LANES:
        ax.add_patch(FancyBboxPatch((0.1, y - 0.25), 16.2, 2.15, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc=color, ec="#cccccc", lw=1))
        ax.text(0.25, y + 1.68, title, fontsize=11, weight="bold", color="#333")
    for name, (x, y, w, h, label) in BOXES.items():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="white", ec="#c0392b" if name in {"pods", "pkg", "mlflow"} else "#2c3e50", lw=1.6))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.6)
    for a, b in ARROWS:
        ax.add_patch(FancyArrowPatch(center(a, "r"), center(b, "l"), arrowstyle="-|>", mutation_scale=13,
                                     color="#555", lw=1.3))
    # cross-lane flows
    for (a, sa), (b, sb), label in [
        (("pkg", "b"), ("arts", "t"), "model artefact"),
        (("arts", "b"), ("logs", "t"), ""),
        (("build", "b"), ("pods", "t"), "image cardiorisk-api:1.0.0"),
    ]:
        p, q = center(a, sa), center(b, sb)
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, color="#c0392b", lw=1.4, ls="--"))
        if label:
            ax.text((p[0] + q[0]) / 2 + 0.1, (p[1] + q[1]) / 2, label, fontsize=8, color="#c0392b")
    ax.add_patch(FancyArrowPatch(center("pods", "b"), (15.2, 2.55), connectionstyle="arc3,rad=0.25",
                                 arrowstyle="-|>", mutation_scale=12, color="#16a085", lw=1.2))
    ax.text(10.5, 2.0, "structured request/prediction logs", fontsize=8, color="#16a085")
    ax.set_title("CardioRisk — end-to-end MLOps architecture", fontsize=14, weight="bold")
    OUT.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
