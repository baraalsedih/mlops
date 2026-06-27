# DVC Demo — Large / On-Prem Dataset Versioning

## Mental model

| Tool | Tracks |
|------|--------|
| **git** | Code, configs, DVC pointer files |
| **DVC** | Large data files (pointers in git, actual bytes on your server) |
| **W&B** | Experiment results, metrics, model artifacts |

Raw data **never leaves your server**. Git only stores a `.dvc` pointer file (an MD5 hash + size). DVC fetches the actual bytes from your configured remote when needed.

---

## Setup

```bash
pip install dvc
git init        # DVC needs a git repo
dvc init

# Point DVC at your on-prem storage (NFS, S3-compatible, SFTP, etc.)
dvc remote add -d onprem /data/mlops-storage
# or for S3-compatible:
# dvc remote add -d onprem s3://your-bucket/dvc-store
```

## Daily workflow

```bash
# 1. Track a new dataset version
dvc add data/wine_raw.csv        # creates data/wine_raw.csv.dvc
git add data/wine_raw.csv.dvc    # commit the pointer, not the data
git commit -m "dataset: add wine_raw v1.0"
dvc push                         # upload bytes to remote

# 2. Run the full pipeline
dvc repro                        # runs prepare → train → evaluate (cached if unchanged)

# 3. Compare experiments
dvc metrics show                 # show current metrics/scores.json
dvc params diff HEAD~1           # compare params vs last commit

# 4. Reproduce an old experiment exactly
git checkout <old-commit>
dvc pull                         # fetches the data that commit pointed to
dvc repro
```

## Pipeline stages (dvc.yaml)

```
data/wine_raw.csv
      │
      ▼
  [prepare]  ──── data/train.csv, data/test.csv
      │
      ▼
   [train]   ──── models/model.pkl
      │
      ▼
 [evaluate]  ──── metrics/scores.json
```

DVC only re-runs a stage when its inputs or parameters change (content-addressed caching).

## Key advantage over W&B Artifacts for large data

- Data stays on **your** infrastructure — no third-party upload
- Supports any storage backend: local FS, NFS, S3, GCS, Azure Blob, SSH
- `.dvc` pointer files are tiny and git-friendly
- Full pipeline reproducibility: same code + same data pointer = identical output
