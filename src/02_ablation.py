"""Tahap 3a: ablasi 24 konfigurasi dengan 5-fold stratified pada data latih.

Faktor: vektorisasi (Count/TF-IDF) x lemmatization (tanpa/dengan POS tag)
x ROS (tanpa/dengan) x algoritma (NB/SVM/LR), hyperparameter bawaan.
Data uji tidak disentuh di skrip ini.

Output:
  outputs/tables/ablasi_24_konfigurasi.csv   mean dan std tiap metrik  [ISI-5]
  outputs/tables/ablasi_negasi.csv           konfigurasi terbaik +/- negasi [ISI-5]
  outputs/tables/ablasi_pengaruh_faktor.csv  arah pengaruh tiap faktor [ISI-15]
"""
import itertools

import pandas as pd
from sklearn.model_selection import cross_validate

from common import ALGO_NAMES, TEXT_COLS, ensure_dirs, load_clean, make_cv, make_pipeline, save_table, split

SCORING = ["accuracy", "balanced_accuracy", "f1_macro"]
METRICS = SCORING + ["fit_time"]


def run(X, y, algo, vec, ros):
    res = cross_validate(make_pipeline(algo, vec, ros), X, y, cv=make_cv(), scoring=SCORING, n_jobs=-1)
    row = {}
    for m in SCORING:
        row[f"{m}_mean"] = res[f"test_{m}"].mean()
        row[f"{m}_std"] = res[f"test_{m}"].std()
    row["fit_time_mean"] = res["fit_time"].mean()
    row["fit_time_std"] = res["fit_time"].std()
    return row


def effect(table, factor, on, off):
    """Rata-rata selisih F1-macro pasangan konfigurasi yang hanya beda satu faktor."""
    others = [c for c in ["algoritma", "vektorisasi", "lemmatization", "ros"] if c != factor]
    piv = table.pivot_table(index=others, columns=factor, values="f1_macro_mean")
    diff = piv[on] - piv[off]
    return {
        "faktor": factor,
        "perbandingan": f"{on} - {off}",
        "selisih_f1_rata2": diff.mean(),
        "selisih_f1_min": diff.min(),
        "selisih_f1_max": diff.max(),
        "naik": int((diff > 0.005).sum()),
        "tidak_berubah": int((diff.abs() <= 0.005).sum()),
        "turun": int((diff < -0.005).sum()),
    }


def main():
    ensure_dirs()
    train, _ = split(load_clean())
    y = train["rating_class"]
    print(f"Ablasi pada {len(train)} ulasan latih, 5-fold stratified")

    rows = []
    for algo, vec, lemma, ros in itertools.product(["nb", "svm", "lr"], ["count", "tfidf"],
                                                   [False, True], [False, True]):
        X = train[TEXT_COLS[(lemma, True)]]
        row = {"algoritma": algo, "vektorisasi": vec, "lemmatization": lemma, "ros": ros}
        row.update(run(X, y, algo, vec, ros))
        rows.append(row)
        print(f"  {algo:3s} {vec:5s} lemma={lemma!s:5s} ros={ros!s:5s} "
              f"f1={row['f1_macro_mean']:.4f}±{row['f1_macro_std']:.4f} "
              f"acc={row['accuracy_mean']:.4f} t={row['fit_time_mean']:.2f}s")

    table = pd.DataFrame(rows)
    table.insert(1, "nama_algoritma", table["algoritma"].map(ALGO_NAMES))
    table = table.sort_values("f1_macro_mean", ascending=False).reset_index(drop=True)
    save_table(table.round(4), "ablasi_24_konfigurasi.csv")

    effects = pd.DataFrame([
        effect(table, "vektorisasi", "tfidf", "count"),
        effect(table, "lemmatization", True, False),
        effect(table, "ros", True, False),
    ])

    best = table.iloc[0]
    neg_rows = []
    for keep_neg in (True, False):
        X = train[TEXT_COLS[(bool(best["lemmatization"]), keep_neg)]]
        row = {"algoritma": best["algoritma"], "vektorisasi": best["vektorisasi"],
               "lemmatization": best["lemmatization"], "ros": best["ros"], "negasi_dipertahankan": keep_neg}
        row.update(run(X, y, best["algoritma"], best["vektorisasi"], bool(best["ros"])))
        neg_rows.append(row)
    neg = pd.DataFrame(neg_rows)
    save_table(neg.round(4), "ablasi_negasi.csv")

    d = neg.loc[0, "f1_macro_mean"] - neg.loc[1, "f1_macro_mean"]
    effects.loc[len(effects)] = {"faktor": "negasi", "perbandingan": "dipertahankan - dibuang",
                                 "selisih_f1_rata2": d, "selisih_f1_min": d, "selisih_f1_max": d,
                                 "naik": int(d > 0.005), "tidak_berubah": int(abs(d) <= 0.005),
                                 "turun": int(d < -0.005)}
    effects["arah"] = pd.cut(effects["selisih_f1_rata2"], [-1, -0.005, 0.005, 1],
                             labels=["turun", "tidak berubah", "naik"])
    save_table(effects.round(4), "ablasi_pengaruh_faktor.csv")
    print(effects.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
