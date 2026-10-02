# Dashboard Tableau (Tahap 7)

Sumber data: `outputs/tableau/tableau_reviews.csv` (satu baris per ulasan,
dibuat oleh `src/08_export_tableau.py`). Simpan setiap tampilan sebagai
gambar ke `outputs/figures/tableau_1_ringkasan.png` sampai
`tableau_5_panjang_keyakinan.png` untuk Gambar 4.1-4.5.

Angka di tampilan Ringkasan harus sama dengan
`outputs/tables/tableau_ringkasan_cek.csv`, `oof_ringkasan.csv`, dan
`validasi_precision.csv`.

## Calculated field

| Nama | Rumus |
| --- | --- |
| `Tingkat Kesesuaian` | `1 - AVG(INT([Is Mismatch]))` |
| `Keluhan Tersembunyi?` | `[Mismatch Type] = "keluhan_tersembunyi"` |
| `Tingkat Keluhan Tersembunyi` | `SUM(IIF([Keluhan Tersembunyi?],1,0)) / SUM(IIF([Score] >= 4,1,0))` |
| `Urutan Tier` | `CASE [Confidence Tier] WHEN "tinggi" THEN 1 WHEN "sedang" THEN 2 ELSE 3 END` |

## Lima tampilan

1. **Ringkasan.** KPI: jumlah ulasan, `Tingkat Kesesuaian`, jumlah per
   `Mismatch Type` (diagram batang), dan jumlah kandidat per
   `Confidence Tier`. Tanpa validasi manual, kolom `Human Label` tidak ada di CSV.
2. **Peta rating terhadap prediksi.** Heatmap: Rows `Score` (1-5),
   Columns `Pred`, Color dan Label `CNT(Review Id)`. Tambahkan persentase
   per baris (Quick Table Calculation, Percent of Total, Compute Using
   Table Across).
3. **Antrean prioritas.** Tabel teks dengan filter
   `Mismatch Type = keluhan_tersembunyi`; kolom `Content`, `Score`,
   `Proba Neg`, `Confidence Tier`, `App Id`, `Review Date`. Kolom `Is Priority`
   (keluhan tersembunyi berkeyakinan sedang atau tinggi, 126 ulasan) bisa dipakai sebagai filter.
   Urutkan menurun menurut `Proba Neg`. Filter interaktif: `App Id`,
   `Confidence Tier`.
4. **Mismatch per aplikasi.** Diagram batang `Tingkat Keluhan
   Tersembunyi` per `App Id`; tampilan kedua berupa garis per
   `Review Month` (atau `App Version`) jika kolomnya terisi.
5. **Panjang ulasan terhadap keyakinan.** Scatter `N Char` (sumbu x,
   skala log) terhadap `Confidence`, warna `Is Mismatch`, dengan trend
   line. Cantumkan koefisien Spearman dari `oof_panjang_vs_keyakinan.csv`
   di judul atau caption.
