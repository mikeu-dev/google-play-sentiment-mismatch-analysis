"""Tahap 7: satu CSV untuk seluruh tampilan Tableau.

Satu baris per ulasan, gabungan prediksi OOF (04), label manual (06), dan
balasan pengembang (07, jika ada). Nama dan foto pengguna tidak diekspor.

Output:
  outputs/tableau/tableau_reviews.csv
  outputs/tables/tableau_ringkasan_cek.csv   angka tampilan Ringkasan; harus sama dengan [ISI-9]/[ISI-10]
  outputs/tables/antrean_prioritas.csv      126 kandidat keluhan tersembunyi berkeyakinan sedang/tinggi (Tabel 4.6)
Petunjuk membangun tampilan Tableau ada di docs/tableau_dashboard.md.
"""
import re

import pandas as pd

from common import ROOT, TABLEAU, TABLES, VALIDATED_CSV, ensure_dirs, load_oof, save_table

COLS = ["review_id", "content", "score", "rating_class", "pred", "proba_neg", "proba_neu", "proba_pos",
        "confidence", "confidence_tier", "is_mismatch", "mismatch_type", "n_char", "n_token",
        "human_label", "is_validated", "validation_group", "has_reply",
        "app_id", "review_date", "review_month", "app_version", "thumbs_up", "split", "model"]


def sensor(text, n=400):
    """Hapus alamat email, tautan, dan nomor telepon dari teks ulasan, lalu potong."""
    text = re.sub(r"\S+@\S+|https?://\S+|\+?\d[\d\s-]{7,}\d", "[dihapus]", str(text))
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "\u2026"


def main():
    ensure_dirs()
    df = load_oof()

    if VALIDATED_CSV.exists():
        v = pd.read_csv(VALIDATED_CSV).rename(columns={"kelompok": "validation_group"})
        df = df.merge(v, on="review_id", how="left")
    else:
        print("Label manual belum ada (06_validasi.py); kolom human_label dikosongkan.")
        df["human_label"], df["validation_group"] = pd.NA, pd.NA
    df["is_validated"] = df["human_label"].notna()

    if "replyContent" in df.columns:
        df["has_reply"] = df["replyContent"].notna() & (df["replyContent"].astype(str).str.strip() != "")
    if "review_date" in df.columns:
        df["review_month"] = pd.to_datetime(df["review_date"], errors="coerce").dt.to_period("M").astype(str)

    if not df["is_validated"].any():
        # Tanpa validasi manual, kolom label dikosongkan agar tidak membingungkan di Tableau
        df = df.drop(columns=["human_label", "is_validated", "validation_group"])
    df["star_group"] = df["score"].map(lambda s: "1-2" if s <= 2 else "3" if s == 3 else "4-5")
    df["confidence_tier_order"] = df["confidence_tier"].map({"tinggi": 1, "sedang": 2, "rendah": 3})
    df["is_priority"] = df["mismatch_type"].eq("keluhan_tersembunyi") & df["confidence_tier"].isin(["tinggi", "sedang"])
    out = df[[c for c in COLS + ["star_group", "confidence_tier_order", "is_priority"] if c in df.columns]]
    path = TABLEAU / "tableau_reviews.csv"
    out.to_csv(path, index=False)
    print(f"  -> {path.relative_to(ROOT)} ({len(out)} baris, {len(out.columns)} kolom)")

    check = [("Jumlah ulasan", len(out)),
             ("Tingkat kesesuaian (%)", round(100 * (1 - out["is_mismatch"].mean()), 2))]
    check += [(f"Jumlah {t}", int(n)) for t, n in out["mismatch_type"].value_counts().items()]
    prec = TABLES / "validasi_precision.csv"
    if prec.exists():
        for _, r in pd.read_csv(prec).iterrows():
            check.append((f"Validasi {r['kelompok']} (%)", round(100 * r["proporsi"], 2)))
    save_table(pd.DataFrame(check, columns=["ukuran", "nilai"]), "tableau_ringkasan_cek.csv")

    q = out[out["is_priority"]].sort_values("proba_neg", ascending=False).reset_index(drop=True)
    antre = pd.DataFrame({
        "peringkat": range(1, len(q) + 1), "review_id": q["review_id"], "app_id": q.get("app_id"),
        "bintang": q["score"], "proba_negatif": q["proba_neg"].map("{:.2f}".format),
        "tingkat_keyakinan": q["confidence_tier"], "tanggal": q["review_date"].astype(str).str[:10],
        "ulasan": q["content"].map(sensor)})
    save_table(antre, "antrean_prioritas.csv")


if __name__ == "__main__":
    main()
