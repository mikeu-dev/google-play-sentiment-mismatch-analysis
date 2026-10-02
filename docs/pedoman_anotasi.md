# Pedoman Anotasi Sentimen Ulasan (Tahap 5)

Lampirkan pedoman ini di skripsi. Pelabel kedua membaca pedoman yang sama
sebelum mulai.

## Tugas

Baca teks ulasan di `data/annotation/lembar_anotasi_pelabel1.csv` (pelabel
kedua: `lembar_anotasi_pelabel2.csv`) dan isi kolom `label` dengan sentimen
**dominan** ulasan: `Positive`, `Neutral`, atau `Negative`.

- Jangan mencari rating bintang atau prediksi model. Keduanya sengaja
  disembunyikan dan urutan baris sudah diacak.
- Jangan membuka `data/annotation/kunci_sampel.csv` sebelum selesai.
- Pelabel pertama dan kedua bekerja terpisah, tanpa berdiskusi.

## Definisi label

| Label | Kapan dipakai | Contoh |
| --- | --- | --- |
| Positive | Pengguna puas, memuji, atau merekomendasikan, dan tidak ada keluhan yang lebih menonjol. | "Love this app, helps me stay organized every day." |
| Negative | Ada keluhan, masalah teknis, kekecewaan, atau permintaan perbaikan yang menjadi inti ulasan, walaupun disertai basa-basi positif. | "Great app but it crashes every time I open the calendar. Please fix." |
| Neutral | Tidak ada sikap yang jelas: pertanyaan, deskripsi, saran ringan tanpa keluhan, atau pujian dan keluhan yang benar-benar seimbang. | "Does this sync with Google Calendar?" |

## Aturan untuk kasus sulit

1. **Campuran pujian dan keluhan.** Pilih sisi yang menjadi alasan utama
   ulasan ditulis. Pola "bagus, tapi X rusak, tolong perbaiki" diberi label
   Negative. Tulis `campuran` di kolom `catatan`.
2. **Permintaan fitur.** Permintaan sopan tanpa nada kecewa diberi Neutral;
   bila disertai kekecewaan ("useless without X"), diberi Negative.
3. **Sarkasme.** Label menurut makna sebenarnya, lalu tulis `sarkasme` di
   catatan.
4. **Teks terlalu pendek atau tidak bermakna.** Pilih Neutral dan tulis
   `tidak jelas` di catatan.
5. **Bahasa selain Inggris.** Label bila Anda paham maknanya; jika tidak,
   kosongkan label dan tulis `bahasa` di catatan.

## Setelah selesai

Simpan file CSV dengan nama yang sama, lalu jalankan
`python src/06_validasi.py`. Commit kedua lembar anotasi ke repositori:
label manual tidak bisa dibuat ulang oleh skrip.
