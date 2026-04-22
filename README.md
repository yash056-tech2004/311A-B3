# Experiment 4: Reproducibility Using Python and Git

This repository contains a small reproducible machine learning experiment on the Iris dataset. The training workflow uses a YAML configuration file, fixed random seeds, dataset hashing, persisted model artifacts, and tracked metrics to make results repeatable.

## Reproduce

```bash
git clone https://github.com/<your-username>/311A-B3.git
cd 311A-B3
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
```
