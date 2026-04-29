# Machine-Learning-CPS5008-SEM2-2025-2026
This project predicts whether a customer support email is likely to escalate to a formal complaint or regulatory review within 14 days.

## Project Overview

- Dataset: customer support emails with text, customer metadata, sentiment, and escalation outcome.
- Prediction task: binary classification where `escalated = (escalation_level >= 2)`.
- Main objective: maximise F2 score (higher weight on recall to reduce missed escalations).
- Selected final model: Logistic Regression, because it achieved the best tuned F2 in this repository and remains interpretable for reporting and stakeholder review.

## Requirements

- Python 3.10+ (developed and tested on Python 3.13)
- A virtual environment (recommended)

## Setup

From the project root:

#Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.lock.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

### Windows Git Bash

#bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.lock.txt
```

## Data Placement

Place the dataset at:

```text
data/customer_support_emails.csv
```

The scripts expect this relative path when run from the project root.

## Run Commands

Run from project root after activating the virtual environment.

```bash
python src/stage2_eda/run.py
python src/stage3_preprocessing/run.py
python src/stage4_models/run.py
python src/stage5_error_analysis/run.py
python src/stage6_interpretability/run.py
python src/stage7_business_recs/run.py
```

Supplementary analysis scripts used in this repo:

```bash
python src/stage3b_feature_engineering/run.py
python src/stage4b_supplementary_models/run.py
```

## Output Files

- Each stage writes CSV/JSON/PNG outputs to its matching folder under `outputs/`.
- Core model comparison artefacts are in `outputs/stage4/`.
- Error analysis artefacts are in `outputs/stage5/`.
- Interpretability and fairness artefacts are in `outputs/stage6/`.
- Business recommendation artefacts are in `outputs/stage7/`.

Note on `outputs/stage4b/linear_regression_baseline.csv`:

- This file is a sanity-check output from supplementary analysis only.
- It is included to show why linear regression is not appropriate for the final binary classifier.
- It was not used for final model selection.

## Reproducibility Notes

- Dependency lock file: `requirements.lock.txt`
- Global randomness control: `random_state=42` used consistently across CV/model components where applicable.
- Validation strategy: `StratifiedGroupKFold` grouped by `customer_id`.
- Primary optimisation/evaluation metric: F2 score.
- Leakage-sensitive fields excluded from modelling features include `resolution_time`, `manual_annotation`, `customer_id`, `timestamp`, `escalation_level`, `escalated`, and raw `email_body_text`.

## Repository Structure

Source code is organised by stage under `src/`, shared helpers are in `src/utils/`, and generated artefacts are under `outputs/`.


