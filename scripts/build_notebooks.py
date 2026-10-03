"""Generate the three project notebooks (EDA, training, inference) from code cells.

Keeping notebooks generated means their logic lives in the tested package and the
notebooks stay thin, readable walkthroughs. Execute them with:
    jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
"""

from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
SETUP = """import sys, pathlib
ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
%matplotlib inline"""


def nb(cells):
    book = nbf.v4.new_notebook()
    book.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    book.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1])
                  for c in cells]
    return book


EDA = [
    ("md", "# 01 · Exploratory Data Analysis — UCI Heart Disease (Cleveland)\n\n"
           "Goal: understand data quality, class balance and which clinical signals separate "
           "patients with and without heart disease, to inform cleaning and feature engineering."),
    ("code", SETUP),
    ("code", "from cardiorisk import config\nfrom cardiorisk.data import prepare, load_raw, load_clean, missing_report\n"
             "clean = prepare()\nraw = load_raw()\nprint(raw.shape, '->', clean.shape)\nraw.head()"),
    ("md", "## Data quality\nMissing values are encoded as `?` in the raw file. Only `ca` (4 rows) and "
           "`thal` (2 rows) are affected. They are **kept as NaN** and imputed *inside* the model "
           "pipeline so imputation statistics are learned on training folds only."),
    ("code", "print(missing_report(raw))\nclean.describe().T.round(2)"),
    ("code", "from IPython.display import Image, display\nfrom cardiorisk.eda import run_all\n"
             "paths = run_all(raw, clean)\nfor p in paths: display(Image(filename=str(p)))"),
    ("md", "## Target\n`num` 0 = no disease, 1–4 = increasing severity. We collapse 1–4 into a single "
           "positive class (the standard binary formulation). The classes are close to balanced "
           "(≈46 % positive), so accuracy is meaningful, but we still report precision/recall/ROC-AUC."),
    ("code", "clean[config.TARGET].value_counts(normalize=True).round(3)"),
    ("md", "## Strongest univariate signals"),
    ("code", "corr = clean.astype(float).corr(method='spearman')[config.TARGET].drop(config.TARGET)\n"
             "corr.reindex(corr.abs().sort_values(ascending=False).index).round(3)"),
    ("md", "## Takeaways that drive modelling\n"
           "* `thal`, `ca`, `cp`, `oldpeak`, `exang` and `thalach` carry most of the signal; `chol` and `fbs` are weak.\n"
           "* `cp`, `thal`, `slope`, `restecg` are **nominal** codes → one-hot encode, never treat as numbers.\n"
           "* `thalach` falls with age, so a raw comparison is confounded. Engineered feature "
           "`hr_reserve_pct = thalach / (220 − age)` measures heart-rate response relative to the age-predicted maximum.\n"
           "* Numeric scales differ by two orders of magnitude (oldpeak vs chol) → standardise for linear models.\n"
           "* A few cholesterol/BP outliers are physiologically plausible, so they are kept (tree models are robust; "
           "the linear model sees them after scaling)."),
]

TRAIN = [
    ("md", "# 02 · Feature engineering, model training & tracking\n\n"
           "Runs the same `cardiorisk.train` module that CI runs, then inspects the MLflow results."),
    ("code", SETUP),
    ("code", "import logging; logging.basicConfig(level=logging.INFO)\nfrom cardiorisk.train import train, candidate_models\n"
             "for name, (est, grid) in candidate_models().items():\n    print(name, {k.replace('model__',''): v for k, v in grid.items()})"),
    ("code", "out = train()\nout['winner']"),
    ("code", "import pandas as pd\nfrom cardiorisk import config\n"
             "pd.read_csv(config.FIGURES_DIR / 'model_comparison.csv', index_col=0).T"),
    ("code", "from IPython.display import Image\nImage(filename=str(config.FIGURES_DIR / '07_model_comparison.png'))"),
    ("md", "## Querying MLflow programmatically"),
    ("code", "import mlflow\nmlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)\n"
             "runs = mlflow.search_runs(experiment_names=[config.MLFLOW_EXPERIMENT],\n"
             "                          filter_string=\"tags.run_type = 'family_best'\", order_by=['metrics.cv_roc_auc_mean DESC'])\n"
             "runs[['tags.mlflow.runName','metrics.cv_roc_auc_mean','metrics.cv_roc_auc_std','metrics.test_roc_auc','metrics.test_recall']].head(3)"),
    ("code", "cands = mlflow.search_runs(experiment_names=[config.MLFLOW_EXPERIMENT], filter_string=\"tags.run_type = 'grid_candidate'\")\n"
             "print(len(cands), 'grid-candidate runs logged')\n"
             "cands.groupby('tags.model_family')['metrics.cv_roc_auc_mean'].describe().round(3)"),
    ("code", "import json; print(json.dumps(json.loads((config.MODELS_DIR/'metadata.json').read_text()), indent=2))"),
]

INFER = [
    ("md", "# 03 · Inference\n\nLoad the packaged model exactly as the API does and score new patients."),
    ("code", SETUP),
    ("code", "from cardiorisk.predict import RiskModel\nmodel = RiskModel()\nprint(model.name, model.version)"),
    ("code", "patients = [\n"
             "  dict(age=58, sex=1, cp=4, trestbps=140, chol=260, fbs=0, restecg=2, thalach=120, exang=1, oldpeak=2.4, slope=2, ca=2, thal=7),\n"
             "  dict(age=41, sex=0, cp=2, trestbps=126, chol=306, fbs=0, restecg=0, thalach=163, exang=0, oldpeak=0.0, slope=1, ca=0, thal=3),\n"
             "  dict(age=63, sex=1, cp=3, trestbps=145, chol=233, fbs=1, restecg=2, thalach=150, exang=0, oldpeak=2.3, slope=3, ca=None, thal=None),\n"
             "]\nimport pandas as pd\npd.DataFrame([p.__dict__ for p in model.predict(patients)])"),
    ("md", "The third patient has unknown `ca`/`thal`: the pipeline's imputers fill them with the training "
           "mode, so the model still returns a prediction. The same request shape is accepted by `POST /predict`."),
    ("code", "import mlflow.pyfunc\nfrom cardiorisk import config\n"
             "pyfunc = mlflow.pyfunc.load_model(str(config.MODELS_DIR / 'mlflow_model'))\n"
             "pyfunc.predict(pd.DataFrame(patients[:2]).astype(float))"),
]

if __name__ == "__main__":
    NB_DIR.mkdir(exist_ok=True)
    for name, cells in [("01_eda.ipynb", EDA), ("02_training.ipynb", TRAIN), ("03_inference.ipynb", INFER)]:
        nbf.write(nb(cells), NB_DIR / name)
        print("wrote", NB_DIR / name)
