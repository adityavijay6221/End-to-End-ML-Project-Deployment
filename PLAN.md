# Deployment Plan: GitHub → CodePipeline → Elastic Beanstalk

## Overview
Every push to `main` auto-deploys to AWS Elastic Beanstalk via CodePipeline.

---

## Step 1 — Fix `requirements.txt` ✅
Replaced `-e .` with `.` (non-editable local install). Editable installs fail on EB.

## Step 2 — Create `.ebignore` ✅
Excludes `venv/`, `logs/`, `data/`, `__pycache__/` from the EB bundle.
`artifacts/` is intentionally included — model and preprocessor pkl files are required at runtime.

## Step 3 — Commit and push ✅
```bash
git add requirements.txt .ebignore
git commit -m "fix requirements and add ebignore for EB deployment"
git push
```

---

## Step 4 — Create EB Application (AWS Console) — manual
1. Go to **Elastic Beanstalk** → **Create application**
2. Name: `mlops-student-performance`
3. Platform: **Python 3.11** (EB does not support Python 3.14)
4. Code: select **Sample application** (CodePipeline will overwrite)
5. Create environment

## Step 5 — Create CodePipeline — manual
1. Go to **CodePipeline** → **Create pipeline**
2. **Source**: GitHub (Version 2) → connect repo → `main` branch → GitHub webhooks
3. **Build**: skip
4. **Deploy**: AWS Elastic Beanstalk → select app + environment from Step 4
5. Create pipeline — first deploy runs immediately

## Step 6 — Verify
- CodePipeline: both stages green (Source → Deploy)
- EB environment health: **Green**
- Open environment URL → `/predict` → submit form → returns prediction

---

## Critical files that must stay in the bundle
| File | Reason |
|---|---|
| `artifacts/preprocessor.pkl` | Fitted transformer required at runtime |
| `artifacts/model.pkl` | Trained model required at runtime |
| `.ebextensions/python.config` | Sets `WSGIPath: app:application` |
| `setup.py` + `src/` | App imports `src.*` |
