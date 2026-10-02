# Audit Pipeline Lama (Tahap 1)

Hasil pemeriksaan `analisis_komparatif_sentimen_dan_mismatch_rating.ipynb`
terhadap daftar periksa Tahap 1 dan 2. Notebook tidak diubah dan disimpan
sebagai arsip; semua angka baru berasal dari skrip `src/01`-`src/08`.

| Butir periksa | Temuan di notebook | Akibat | Perbaikan di `src/` |
| --- | --- | --- | --- |
| ROS sebelum `train_test_split`? | Tidak. `fit_resample` dipanggil pada `X_train` saja. | Aman. | ROS dipindah ke dalam `imblearn.Pipeline` agar juga aman di tiap fold CV. |
| TF-IDF `fit_transform` sebelum pembagian? | **Ya.** `vectorizer.fit_transform(df['clean_review'])` dijalankan pada seluruh 12.495 ulasan, baru kemudian dibagi. | Kosakata dan bobot IDF ikut belajar dari data uji (kebocoran). Akurasi 67,71% / 66,15% / 66,11% tidak sah. | TF-IDF di dalam pipeline (`common.make_pipeline`). |
| Duplikat teks identik dihapus? | Tidak. | Ulasan seperti "good app" bisa ada di data latih dan uji sekaligus. | `01_preprocess.py` menghapus duplikat `reviewId` dan duplikat teks bersih, lalu mencatat jumlahnya. |
| `stratify` dan `random_state`? | Ya (`stratify=y`, `random_state=42`). | Aman. | Dipertahankan; satu konstanta `SEED`. |
| `GridSearchCV` membungkus pipeline? | **Grid Search dan K-Fold tidak dipakai sama sekali**; `cross_val_score` diimpor tetapi tidak dipanggil. | Klaim 2.3.6 tidak didukung kode. | `02_ablation.py` (5-fold) dan `03_final_models.py` (`GridSearchCV` atas seluruh pipeline). |
| Stopword NLTK | Seluruh daftar dibuang, termasuk "no", "not", "nor". Apostrof dihapus sehingga "don't" menjadi "dont". | "not good" menjadi "good". | Negasi dipertahankan dan bentuk n't dinormalisasi. |
| Lemmatization | `WordNetLemmatizer` tanpa POS tag. | Semua kata dianggap kata benda; "crashed" tetap "crashed". | Lemmatization dengan POS tag. |
| Regex pembersih | `[^a-z\s]` diganti string kosong. | "app.great" menjadi "appgreat". | Diganti spasi. |
| Ulasan kosong setelah dibersihkan | Tidak ditandai. | Model hanya menebak untuk ulasan emoji/angka. | Kolom `n_token`; `n_token == 0` dikeluarkan dan dicatat. |
| Mismatch dihitung pada data apa? | Ekspor hanya memuat data uji (`df.iloc[y_test.index]`), padahal naskah menyebut 12.495 data. | Angka 68,39% / 31,61% tidak bisa dilacak ke kode, dan prediksi pada data latih tidak bermakna sebagai "mismatch". | `04_oof_mismatch.py`: prediksi out-of-fold untuk semua ulasan. |
| `df.iloc[y_test.index]` | `y_test.index` adalah label indeks, sedangkan `iloc` memakai posisi. | Bila ada baris yang terbuang oleh `dropna`, teks dan prediksi tidak sejajar. | Semua penggabungan memakai `review_id`. |
| Confidence score | `LinearSVC` tidak punya `predict_proba`; `tableau_export.py` mengisi `confidence_score = 1.0` bila probabilitas tidak ada. | Analisis panjang ulasan terhadap keyakinan (3.1.5) tidak bisa dilakukan. | `04` menyimpan `proba_neg/neu/pos`; SVM dikalibrasi bila dipilih. |
