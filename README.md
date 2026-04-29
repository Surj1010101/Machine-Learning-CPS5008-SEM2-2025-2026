# Machine-Learning-CPS5008-SEM2-2025-2026

CPS5008 Assessment 2 — Customer Support Email Escalation Prediction.

Predicts whether a customer support email to a national energy company will escalate to a formal complaint or regulatory review within 14 days.

---

## Requirements

- Python **3.10 or newer** (developed and tested on Python 3.13)
- Approx. 500 MB disk space for the virtual environment

---

## Setup (first time, any machine)

From the project root:

### Windows (Git Bash)
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.lock.txt
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.lock.txt
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```


# Running the pipeline

Always activate the virtual environment first (see Setup), then:

### Stage 2 — Exploratory Data Analysis & Risk Identification
```bash
python src/stage2_eda/run.py
```
Outputs:
- Console analysis (dataset shape, class imbalance, missing data, leakage tests, chi-square results, segment imbalance, customer overlap, numeric correlations)
- `outputs/stage2/eda_overview.png`
- `outputs/stage2/correlation_heatmap.png`
- `outputs/stage2/escalation_by_region.png`


#Project structure


