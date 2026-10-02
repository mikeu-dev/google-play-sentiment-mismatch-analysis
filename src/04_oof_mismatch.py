"""Tahap 4: mismatch dengan prediksi out-of-fold (OOF).

Setiap ulasan (latih dan uji) diprediksi oleh model yang tidak melihatnya
saat pelatihan, lewat 5-fold ``cross_val_predict``. Tingkat kesesuaian di
sini sama dengan akurasi OOF, dan itu wajar: yang menjadikannya temuan
adalah validasi manual di Tahap 5.

Model default: Logistic Regression dengan konfigurasi dari 03, karena
menghasilkan probabilitas. Ganti dengan ``--algo nb`` atau ``--algo svm``
(SVM dikalibrasi dengan CalibratedClassifierCV agar punya probabilitas).

Output:
  data/processed/oof_predictions.csv
  outputs/tables/oof_ringkasan.csv           tingkat kesesuaian, angka utama  [ISI-9]
  outputs/tables/oof_tipe_mismatch.csv       jumlah per tipe x tingkat keyakinan [ISI-9]
  outputs/tables/oof_rating_vs_prediksi.csv  crosstab bintang 1-5 x prediksi
  outputs/tables/oof_panjang_vs_keyakinan.csv korelasi panjang ulasan dan confidence
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import cross_val_predict

from common import (ALGO_NAMES, BEST_JSON, CLASSES, OOF_CSV, ROOT, TEXT_COLS, TIER_HIGH, TIER_MED,
                    ensure_dirs, load_clean, load_json, make_cv, make_pipeline, save_table)

MISMATCH_ORDER = ["sesuai", "keluhan_tersembunyi", "pujian_tersembunyi", "rating3_tidak_netral", "lainnya"]


def mismatch_type(score, pred):
    if score >= 4 and pred == "negative":
        return "keluhan_tersembunyi"
    if score <= 2 and pred == "positive":
        return "pujian_tersembunyi"
    if score == 3 and pred != "neutral":
        return "rating3_tidak_netral"
    if score != 3 and pred == "neutral":
        return "lainnya"
    return "sesuai"


def confidence_tier(c):
    return "tinggi" if c >= TIER_HIGH else "sedang" if c >= TIER_MED else "rendah"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="lr", choices=["nb", "svm", "lr"])
    args = ap.parse_args()

    ensure_dirs()
    if not BEST_JSON.exists():
        raise SystemExit(f"{BEST_JSON} belum ada. Jalankan src/03_final_models.py lebih dulu.")
    cfg = load_json(BEST_JSON)["configs"][args.algo]
    df = load_clean()
    X = df[TEXT_COLS[(cfg["lemmatization"], True)]]
    y = df["rating_class"].values

    pipe = make_pipeline(args.algo, cfg["vektorisasi"], cfg["ros"], calibrate=args.algo == "svm")
    if args.algo == "svm":
        pipe.set_params(**{f"clf__estimator__{k}": v for k, v in cfg["params"].items()})
    else:
        pipe.set_params(**{f"clf__{k}": v for k, v in cfg["params"].items()})
    print(f"OOF {ALGO_NAMES[args.algo]} {cfg} pada {len(df)} ulasan")

    proba = cross_val_predict(pipe, X, y, cv=make_cv(), method="predict_proba", n_jobs=-1)
    kelas = np.array(sorted(np.unique(y)))     # urutan kolom predict_proba
    assert list(kelas) == CLASSES

    out = df.drop(columns=[c for c in TEXT_COLS.values() if c != TEXT_COLS[(True, True)]])
    out = out.rename(columns={TEXT_COLS[(True, True)]: "clean_text"})
    out["model"] = args.algo
    out["pred"] = kelas[proba.argmax(axis=1)]
    out["proba_neg"], out["proba_neu"], out["proba_pos"] = proba[:, 0], proba[:, 1], proba[:, 2]
    out["confidence"] = proba.max(axis=1)
    out["confidence_tier"] = out["confidence"].apply(confidence_tier)
    out["is_mismatch"] = out["pred"] != out["rating_class"]
    out["mismatch_type"] = [mismatch_type(s, p) for s, p in zip(out["score"], out["pred"])]
    out.to_csv(OOF_CSV, index=False)
    print(f"  -> {OOF_CSV.relative_to(ROOT)}")

    hi = out[out["score"] >= 4]
    hidden = hi[hi["pred"] == "negative"]
    summary = pd.DataFrame([
        ("Model", ALGO_NAMES[args.algo]),
        ("Jumlah ulasan dianalisis", len(out)),
        ("Tingkat kesesuaian OOF (%)", round(100 * (1 - out["is_mismatch"].mean()), 2)),
        ("Mismatch OOF (%)", round(100 * out["is_mismatch"].mean(), 2)),
        ("Balanced accuracy OOF", round(balanced_accuracy_score(y, out["pred"]), 4)),
        ("F1-macro OOF", round(f1_score(y, out["pred"], average="macro"), 4)),
        ("Ulasan bintang 4-5", len(hi)),
        ("Bintang 4-5 diprediksi negative (keluhan tersembunyi)", len(hidden)),
        ("Keluhan tersembunyi (% dari bintang 4-5)", round(100 * len(hidden) / max(len(hi), 1), 2)),
        ("Keluhan tersembunyi, keyakinan tinggi", int((hidden["confidence_tier"] == "tinggi").sum())),
        ("Akurasi OOF (cek)", round(accuracy_score(y, out["pred"]), 4)),
    ], columns=["ukuran", "nilai"])
    save_table(summary, "oof_ringkasan.csv")
    print(summary.to_string(index=False))

    tipe = pd.crosstab(out["mismatch_type"], out["confidence_tier"], margins=True, margins_name="total")
    tipe = tipe.reindex(index=MISMATCH_ORDER + ["total"],
                        columns=["tinggi", "sedang", "rendah", "total"], fill_value=0)
    tipe["persen_dari_semua"] = (tipe["total"] / len(out) * 100).round(2)
    save_table(tipe.reset_index(), "oof_tipe_mismatch.csv")

    save_table(pd.crosstab(out["score"], out["pred"]).reindex(columns=CLASSES, fill_value=0).reset_index(),
               "oof_rating_vs_prediksi.csv")

    rows = []
    for name, sub in [("semua", out), ("mismatch", out[out["is_mismatch"]]), ("sesuai", out[~out["is_mismatch"]])]:
        for col in ["n_char", "n_token"]:
            rho, p = spearmanr(sub[col], sub["confidence"])
            rows.append({"kelompok": name, "variabel": col, "n": len(sub), "spearman_rho": rho, "p_value": p})
    save_table(pd.DataFrame(rows).round(4), "oof_panjang_vs_keyakinan.csv")


if __name__ == "__main__":
    main()
