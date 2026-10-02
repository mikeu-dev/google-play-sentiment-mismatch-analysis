"""Tahap 5b: hitung precision validasi manual, perkiraan jumlah nyata, dan kappa.

Membaca lembar anotasi yang sudah diisi (05) dan kuncinya.

Output:
  outputs/tables/validasi_precision.csv            precision per kelompok, CI Wilson 95%  [ISI-10]
  outputs/tables/validasi_precision_per_tier.csv   per tingkat keyakinan                  [ISI-10]
  outputs/tables/validasi_perkiraan_jumlah.csv     kandidat x precision                   [ISI-10]
  outputs/tables/validasi_kappa.csv                Cohen's kappa dua pelabel              [ISI-10]
  outputs/tables/validasi_contoh.csv               contoh per kelompok untuk Tabel 4.6   [ISI-13]
  data/processed/validated_labels.csv              human_label per review_id (untuk 08)

Definisi "benar" per kelompok:
  keluhan_tersembunyi   label manusia Negative (manusia setuju dengan model)
  pujian_tersembunyi    label manusia Positive
  rating3_tidak_netral  label manusia bukan Neutral (rating 3 memang tidak netral);
                        kolom tambahan: label manusia sama dengan prediksi model
  kontrol               proporsi label Negative = keluhan yang terlewat model
"""
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.proportion import proportion_confint

from common import ANNOTATION, ROOT, VALIDATED_CSV, ensure_dirs, load_oof, save_table

KEY = ANNOTATION / "kunci_sampel.csv"
SHEET1 = ANNOTATION / "lembar_anotasi_pelabel1.csv"
SHEET2 = ANNOTATION / "lembar_anotasi_pelabel2.csv"
VALID = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative",
         "pos": "Positive", "neu": "Neutral", "neg": "Negative"}

HIT = {
    "keluhan_tersembunyi": lambda d: d["human_label"] == "Negative",
    "pujian_tersembunyi": lambda d: d["human_label"] == "Positive",
    "rating3_tidak_netral": lambda d: d["human_label"] != "Neutral",
    "kontrol": lambda d: d["human_label"] == "Negative",
}
UKURAN = {
    "keluhan_tersembunyi": "precision (manusia: Negative)",
    "pujian_tersembunyi": "precision (manusia: Positive)",
    "rating3_tidak_netral": "proporsi rating 3 yang tidak netral",
    "kontrol": "proporsi keluhan yang terlewat (manusia: Negative)",
}
POP = {
    "keluhan_tersembunyi": lambda d: (d["score"] >= 4) & (d["pred"] == "negative"),
    "pujian_tersembunyi": lambda d: (d["score"] <= 2) & (d["pred"] == "positive"),
    "rating3_tidak_netral": lambda d: (d["score"] == 3) & (d["pred"] != "neutral"),
    "kontrol": lambda d: (d["score"] >= 4) & (d["pred"] == "positive"),
}


def read_sheet(path):
    if not path.exists():
        raise SystemExit(f"{path} tidak ditemukan. Jalankan src/05_sampling_anotasi.py lalu isi labelnya.")
    s = pd.read_csv(path, dtype={"label": str}, keep_default_na=False)
    s["human_label"] = s["label"].str.strip().str.lower().map(VALID)
    bad = s[s["human_label"].isna()]
    if len(bad):
        print(f"Peringatan: {len(bad)} baris di {path.name} belum berlabel atau labelnya tidak dikenali; diabaikan.")
    return s.dropna(subset=["human_label"])


def prop_row(group, sub, hits, **extra):
    k, n = int(hits.sum()), len(sub)
    lo, hi = proportion_confint(k, n, method="wilson") if n else (float("nan"), float("nan"))
    return {"kelompok": group, **extra, "ukuran": UKURAN[group], "n": n, "benar": k,
            "proporsi": k / n if n else float("nan"), "ci95_bawah": lo, "ci95_atas": hi}


def main():
    ensure_dirs()
    key = pd.read_csv(KEY)
    s1 = read_sheet(SHEET1)
    lab = key.merge(s1[["review_id", "human_label", "catatan"]], on="review_id", how="inner")
    print(f"{len(lab)} dari {len(key)} sampel sudah berlabel")

    rows, tier_rows = [], []
    for g, sub in lab.groupby("kelompok"):
        row = prop_row(g, sub, HIT[g](sub))
        if g == "rating3_tidak_netral":
            agree = (sub["human_label"].str.lower() == sub["pred"]).sum()
            row["setuju_dengan_prediksi_model"] = agree / len(sub)
        rows.append(row)
        for tier, t in sub.groupby("confidence_tier"):
            tier_rows.append(prop_row(g, t, HIT[g](t), confidence_tier=tier))
    prec = pd.DataFrame(rows)
    save_table(prec.round(4), "validasi_precision.csv")
    save_table(pd.DataFrame(tier_rows).round(4), "validasi_precision_per_tier.csv")
    print(prec.round(4).to_string(index=False))

    # Perkiraan jumlah mismatch nyata = jumlah kandidat x precision (beserta CI)
    oof = load_oof()
    est = []
    for _, r in prec.iterrows():
        n_pop = int(POP[r["kelompok"]](oof).sum())
        est.append({"kelompok": r["kelompok"], "jumlah_kandidat": n_pop, "proporsi": r["proporsi"],
                    "perkiraan_jumlah": n_pop * r["proporsi"],
                    "perkiraan_bawah": n_pop * r["ci95_bawah"], "perkiraan_atas": n_pop * r["ci95_atas"]})
    est = pd.DataFrame(est).round({"proporsi": 4, "perkiraan_jumlah": 1, "perkiraan_bawah": 1, "perkiraan_atas": 1})
    save_table(est, "validasi_perkiraan_jumlah.csv")

    if SHEET2.exists():
        s2 = read_sheet(SHEET2)
        both = s1[["review_id", "human_label"]].merge(s2[["review_id", "human_label"]],
                                                      on="review_id", suffixes=("_a", "_b"))
        if len(both):
            kappa = cohen_kappa_score(both["human_label_a"], both["human_label_b"])
            agree = (both["human_label_a"] == both["human_label_b"]).mean()
            save_table(pd.DataFrame([{"n_ulasan_ganda": len(both), "kesepakatan_persen": 100 * agree,
                                      "cohen_kappa": kappa}]).round(4), "validasi_kappa.csv")
            print(f"Cohen's kappa = {kappa:.4f} (n={len(both)})")
        else:
            print("Lembar pelabel kedua belum berisi label; kappa belum dihitung.")

    content = oof.set_index("review_id")["content"]
    ex = lab.assign(content=lab["review_id"].map(content))
    ex = ex.sort_values("confidence", ascending=False).groupby("kelompok").head(3)
    save_table(ex[["kelompok", "score", "pred", "confidence", "human_label", "content"]],
               "validasi_contoh.csv")

    lab[["review_id", "kelompok", "human_label"]].to_csv(VALIDATED_CSV, index=False)
    print(f"  -> {VALIDATED_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
