"""Tahap 6: uji asumsi bisnis dengan balasan pengembang (replyContent).

Membandingkan tingkat balasan pada tiga kelompok:
  bintang_1_2              diduga paling sering dibalas
  keluhan_tersembunyi      bintang 4-5, prediksi negative; diduga jarang dibalas
  bintang_4_5_positif      bintang 4-5, prediksi positive; jarang dibalas, memang tidak perlu

Output:
  outputs/tables/balasan_per_kelompok.csv          [ISI-11]
  outputs/tables/balasan_per_aplikasi.csv          [ISI-11]
  outputs/tables/balasan_uji_chi_square.csv        [ISI-11]
Jika kolom replyContent tidak ada, hanya ditulis balasan_tidak_tersedia.csv.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from common import ensure_dirs, load_oof, save_table

ORDER = ["bintang_1_2", "keluhan_tersembunyi", "bintang_4_5_positif"]


def kelompok(row):
    if row["score"] <= 2:
        return "bintang_1_2"
    if row["score"] >= 4 and row["pred"] == "negative":
        return "keluhan_tersembunyi"
    if row["score"] >= 4 and row["pred"] == "positive":
        return "bintang_4_5_positif"
    return None


def chi2(d, a, b):
    sub = d[d["kelompok"].isin([a, b])]
    tab = pd.crosstab(sub["kelompok"], sub["has_reply"])
    if tab.shape != (2, 2):
        return {"perbandingan": f"{a} vs {b}", "chi2": np.nan, "dof": np.nan, "p_value": np.nan}
    c, p, dof, _ = chi2_contingency(tab)
    return {"perbandingan": f"{a} vs {b}", "chi2": c, "dof": dof, "p_value": p}


def main():
    ensure_dirs()
    df = load_oof()
    if "replyContent" not in df.columns:
        save_table(pd.DataFrame([{"catatan": "Kolom replyContent tidak tersedia di dataset; Tahap 6 dilewati."}]),
                   "balasan_tidak_tersedia.csv")
        print("Kolom replyContent tidak ada; Tahap 6 dilewati.")
        return

    df["has_reply"] = df["replyContent"].notna() & (df["replyContent"].astype(str).str.strip() != "")
    df["kelompok"] = df.apply(kelompok, axis=1)
    d = df.dropna(subset=["kelompok"])

    rate = d.groupby("kelompok")["has_reply"].agg(tingkat_balasan="mean", jumlah_dibalas="sum", n="size")
    rate = rate.reindex(ORDER)
    rate["tingkat_balasan_persen"] = (rate["tingkat_balasan"] * 100).round(2)
    save_table(rate.reset_index().round(4), "balasan_per_kelompok.csv")
    print(rate.to_string())

    if "app_id" in d.columns:
        per_app = d.pivot_table(index="app_id", columns="kelompok", values="has_reply",
                                aggfunc=["mean", "size"]).round(4)
        per_app.columns = [f"{'tingkat' if a == 'mean' else 'n'}_{k}" for a, k in per_app.columns]
        save_table(per_app.reset_index(), "balasan_per_aplikasi.csv")

    tab = pd.crosstab(d["kelompok"], d["has_reply"])
    c, p, dof, _ = chi2_contingency(tab)
    tests = [{"perbandingan": "tiga kelompok", "chi2": c, "dof": dof, "p_value": p}]
    tests += [chi2(d, a, b) for a, b in [("keluhan_tersembunyi", "bintang_1_2"),
                                         ("keluhan_tersembunyi", "bintang_4_5_positif")]]
    save_table(pd.DataFrame(tests).round(6), "balasan_uji_chi_square.csv")


if __name__ == "__main__":
    main()
