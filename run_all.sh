#!/usr/bin/env bash
# Menjalankan seluruh pipeline dari awal (tanpa validasi manual).
# Langkah opsional validasi manual (05 dan 06) tidak dijalankan di sini; lihat README.md.
set -euo pipefail
cd "$(dirname "$0")"
python src/01_preprocess.py
python src/02_ablation.py
python src/03_final_models.py
python src/04_oof_mismatch.py              # analisis utama: Logistic Regression
python src/04_oof_mismatch.py --algo svm   # kekokohan: Linear SVM terkalibrasi (berkas berakhiran _svm)
python src/07_reply_rate.py
python src/08_export_tableau.py
