# Analisis Komparatif Algoritma Klasifikasi Sentimen untuk Mengukur Kesesuaian Star Rating dan Ulasan Pengguna

Proyek ini membandingkan Multinomial Naive Bayes, Linear SVM, dan Logistic
Regression untuk menebak sentimen ulasan Google Play dari teksnya, lalu
memakai prediksi *out-of-fold* untuk menemukan ulasan yang rating bintangnya
tidak sesuai dengan isinya. Kandidat *mismatch* divalidasi secara manual.

Dataset: [Google Play Store Reviews](https://www.kaggle.com/datasets/prakharrathi25/google-play-store-reviews)
(12.495 ulasan). Label sentimen diturunkan dari rating: 1-2 negative,
3 neutral, 4-5 positive.

## Struktur

```text
data/raw/reviews.csv           dataset mentah (diunduh otomatis lewat kagglehub bila belum ada)
data/processed/                hasil antara (tidak di-commit)
data/annotation/               lembar anotasi manual (WAJIB di-commit)
src/common.py                  SEED, path, ambang, rentang Grid Search, pipeline
src/preprocessing.py           pembersih teks (negasi, lemmatization dengan POS tag)
src/01_preprocess.py           Tahap 1-2: pembersihan, duplikat, pembagian latih/uji
src/02_ablation.py             Tahap 3: ablasi 24 konfigurasi (5-fold) + negasi
src/03_final_models.py         Tahap 3: Grid Search, evaluasi uji, McNemar, confusion matrix
src/04_oof_mismatch.py         Tahap 4: prediksi out-of-fold dan tipe mismatch
src/05_sampling_anotasi.py     Tahap 5: 400 sampel dan lembar anotasi
src/06_validasi.py             Tahap 5: precision (CI Wilson 95%), perkiraan jumlah, kappa
src/07_reply_rate.py           Tahap 6: tingkat balasan pengembang + chi-square
src/08_export_tableau.py       Tahap 7: CSV untuk Tableau
outputs/tables/  outputs/figures/  outputs/tableau/  outputs/models/
docs/audit_pipeline.md         temuan audit notebook lama (kebocoran TF-IDF, dll.)
docs/pedoman_anotasi.md        pedoman pelabelan manual (lampiran skripsi)
docs/tableau_dashboard.md      cara membangun lima tampilan Tableau
```

Notebook `analisis_komparatif_sentimen_dan_mismatch_rating.ipynb` disimpan
sebagai arsip. Angkanya (67,71%, 68,39%, 31,61%, dst.) tidak dipakai lagi
karena TF-IDF di-*fit* sebelum pembagian data; lihat `docs/audit_pipeline.md`.

## Cara menjalankan

```bash
pip install -r requirements.txt
python src/01_preprocess.py
python src/02_ablation.py
python src/03_final_models.py
python src/04_oof_mismatch.py          # default Logistic Regression; --algo nb|svm
python src/05_sampling_anotasi.py      # lalu isi label di data/annotation/ (lihat docs/pedoman_anotasi.md)
python src/06_validasi.py
python src/07_reply_rate.py
python src/08_export_tableau.py
```

`run_all.sh` menjalankan urutan yang sama. `05` menolak menimpa lembar
anotasi yang sudah ada (pakai `--force` hanya bila ingin mengulang).

Bila NLTK tidak bisa mengunduh data di belakang proxy, set
`NLTK_ALLOW_PROXIED_URLOPEN=1`. Path dataset bisa diganti dengan variabel
lingkungan `REVIEWS_CSV`.

## Keputusan yang dikunci (sama dengan naskah)

| Keputusan | Nilai | Letak di kode | Subbab |
| --- | --- | --- | --- |
| Seed | 42 untuk pembagian data, ROS, CV, sampling | `common.SEED` | 3.2 |
| Pembagian data | 80/20 stratified; ablasi 5-fold pada data latih | `common.TEST_SIZE`, `N_FOLDS` | 3.2.1 |
| Rentang Grid Search | NB `alpha` 0,1/0,5/1,0; SVM dan LR `C` 0,1/1/10 | `common.PARAM_GRID` | 3.2.5 |
| Ambang keyakinan | tinggi >= 0,80; sedang 0,60-0,79; rendah < 0,60 | `common.TIER_HIGH`, `TIER_MED` | 3.4.2 |
| Sampel validasi | 150 / 75 / 75 / 100 (kontrol); 100 untuk pelabel kedua | `common.SAMPLE_SIZES` | 3.4.3 |
| Kolom balasan pengembang | dicek otomatis; lihat `outputs/tables/kolom_dataset.csv` | `07_reply_rate.py` | 3.4.6 |

## Dari skrip ke penanda `[ISI-n]` di naskah

| Penanda | File output |
| --- | --- |
| `[ISI-1]` | `model_akhir.csv` (baris model terbaik) |
| `[ISI-2]` | `oof_ringkasan.csv`, `validasi_precision.csv` |
| `[ISI-3]` | `data_cleaning.csv`, `distribusi_kelas.csv` |
| `[ISI-4]` | `model_akhir.csv` |
| `[ISI-5]` | `ablasi_24_konfigurasi.csv`, `ablasi_negasi.csv` |
| `[ISI-6]` | `mcnemar.csv` |
| `[ISI-7]` | `classification_report_*.csv`, `figures/confusion_matrix_*.png`, `lr_top20_kata.csv`, `figures/lr_top20_kata.png` |
| `[ISI-8]` | `grid_search.csv` |
| `[ISI-9]` | `oof_ringkasan.csv`, `oof_tipe_mismatch.csv`, `oof_rating_vs_prediksi.csv` |
| `[ISI-10]` | `validasi_precision.csv`, `validasi_precision_per_tier.csv`, `validasi_perkiraan_jumlah.csv`, `validasi_kappa.csv` |
| `[ISI-11]` | `balasan_per_kelompok.csv`, `balasan_uji_chi_square.csv` (atau `balasan_tidak_tersedia.csv`) |
| `[ISI-12]` | `outputs/tableau/tableau_reviews.csv` + `docs/tableau_dashboard.md` |
| `[ISI-13]` | `validasi_contoh.csv` |
| `[ISI-15]` | `ablasi_pengaruh_faktor.csv` (selisih F1-macro; ambang "tidak berubah" ±0,005) |

Semua file tabel ada di `outputs/tables/` kecuali disebut lain. Salin angka
ke naskah dari file-file ini, bukan dari layar terminal.
