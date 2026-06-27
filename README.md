# MLOps Training Session — 2.5 Hours

Hands-on materials for the following three topics:

| # | Topic | Duration |
|---|-------|----------|
| 0 | Intro & MLOps landscape | 5 min |
| 1 | W&B: experiment tracking + versioning (+ DVC) | 45 min |
| 2 | CI/CD for ML: Docker + GitHub Actions | 45 min |
| 3 | Production monitoring + drift detection | 45 min |
| 4 | Q&A | 10 min |

---

## Quick Setup

### Prerequisites
- Python 3.10+
- Docker Desktop
- A free [Weights & Biases account](https://wandb.ai)

### 1. Clone and install per section

**Section 1 — W&B + DVC**
```bash
cd 01_wandb
pip install -r requirements.txt
wandb login          # paste your API key from wandb.ai/authorize
```

**Section 2 — CI/CD**
```bash
cd 02_cicd
pip install -r requirements.txt
```

### 2. GitHub Secrets needed for the Actions workflow

| Secret name | Where to get it |
|-------------|----------------|
| `GHCR_TOKEN` | GitHub → Settings → Developer settings → Personal access tokens |
| `WANDB_API_KEY` | https://wandb.ai/authorize |

---

## Structure

```
mlops/
├── 01_wandb/               ← W&B experiment tracking, metadata versioning, DVC
│   ├── train_baseline.py   ← "before" — no tracking
│   ├── train_with_wandb.py ← full W&B integration
│   ├── data_versioning_wandb.py  ← metadata-only artifact (no raw data upload)
│   ├── model_registry.py   ← model artifact + production alias
│   └── dvc_demo/           ← DVC for large/on-prem datasets
│
└── 02_cicd/                ← Docker + GitHub Actions + automated tests
    ├── src/                ← train.py, predict.py
    ├── tests/              ← pytest test suite
    ├── Dockerfile
    ├── docker-compose.yml
    ├── Makefile
    └── .github/workflows/  ← ml-pipeline.yml
```
