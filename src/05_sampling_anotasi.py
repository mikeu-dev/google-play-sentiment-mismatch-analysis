"""Tahap 5a: ambil 400 sampel untuk validasi manual dan buat lembar anotasi.

Lembar anotasi hanya berisi review_id dan teks asli; rating dan prediksi
model disembunyikan dan urutan baris diacak. Kunci (kelompok, rating,
prediksi) disimpan terpisah dan baru dibuka oleh 06_validasi.py.

Output (data/annotation/):
  kunci_sampel.csv              kelompok, rating, prediksi tiap sampel (jangan dibuka saat melabeli)
  lembar_anotasi_pelabel1.csv   400 ulasan; isi kolom ``label`` dan ``catatan``
  lembar_anotasi_pelabel2.csv   100 ulasan yang sama untuk pelabel kedua (Cohen's kappa)

Skrip menolak menimpa lembar yang sudah ada supaya label manual tidak
hilang; pakai ``--force`` hanya jika memang ingin mengulang dari nol.
"""
import argparse

import pandas as pd

from common import ANNOTATION, N_SECOND_ANNOTATOR, ROOT, SAMPLE_SIZES, SEED, ensure_dirs, load_oof, save_table

KEY = ANNOTATION / "kunci_sampel.csv"
SHEET1 = ANNOTATION / "lembar_anotasi_pelabel1.csv"
SHEET2 = ANNOTATION / "lembar_anotasi_pelabel2.csv"


def groups(df):
    return {
        "keluhan_tersembunyi": df[(df["score"] >= 4) & (df["pred"] == "negative")],
        "pujian_tersembunyi": df[(df["score"] <= 2) & (df["pred"] == "positive")],
        "rating3_tidak_netral": df[(df["score"] == 3) & (df["pred"] != "neutral")],
        "kontrol": df[(df["score"] >= 4) & (df["pred"] == "positive")],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="timpa lembar anotasi yang sudah ada")
    args = ap.parse_args()

    ensure_dirs()
    existing = [p for p in (KEY, SHEET1, SHEET2) if p.exists()]
    if existing and not args.force:
        raise SystemExit("Lembar anotasi sudah ada, tidak ditimpa: "
                         + ", ".join(str(p.relative_to(ROOT)) for p in existing))

    df = load_oof()
    parts, info = [], []
    for name, pool in groups(df).items():
        n = min(SAMPLE_SIZES[name], len(pool))
        s = pool.sample(n, random_state=SEED).assign(kelompok=name)
        parts.append(s)
        info.append({"kelompok": name, "populasi": len(pool), "target_sampel": SAMPLE_SIZES[name], "sampel": n})
        if n < SAMPLE_SIZES[name]:
            print(f"Peringatan: kelompok {name} hanya punya {len(pool)} ulasan")
    sample = pd.concat(parts)
    save_table(pd.DataFrame(info), "anotasi_ukuran_sampel.csv")

    cols = ["review_id", "kelompok", "score", "rating_class", "pred", "confidence", "confidence_tier"]
    sample[cols].to_csv(KEY, index=False)

    sheet = sample[["review_id", "content"]].sample(frac=1, random_state=SEED).assign(label="", catatan="")
    sheet.to_csv(SHEET1, index=False)
    sheet.sample(min(N_SECOND_ANNOTATOR, len(sheet)), random_state=SEED + 1).to_csv(SHEET2, index=False)
    for p in (KEY, SHEET1, SHEET2):
        print(f"  -> {p.relative_to(ROOT)}")
    print("Isi kolom label dengan Positive, Neutral, atau Negative mengikuti docs/pedoman_anotasi.md.")


if __name__ == "__main__":
    main()
