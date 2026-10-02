#!/usr/bin/env bash
# Menjalankan Tahap 1-4 dari awal. Tahap 5 butuh pelabelan manual:
# setelah 05 membuat lembar anotasi, isi labelnya, lalu jalankan 06-08.
set -euo pipefail
cd "$(dirname "$0")"
python src/01_preprocess.py
python src/02_ablation.py
python src/03_final_models.py
python src/04_oof_mismatch.py
if [ ! -f data/annotation/lembar_anotasi_pelabel1.csv ]; then
  python src/05_sampling_anotasi.py
  echo "Isi label di data/annotation/, lalu jalankan: python src/06_validasi.py && python src/07_reply_rate.py && python src/08_export_tableau.py"
else
  python src/06_validasi.py
  python src/07_reply_rate.py
  python src/08_export_tableau.py
fi
