"""Tahap 3b: Grid Search, evaluasi tiga model akhir pada data uji, dan McNemar.

Untuk tiap algoritma diambil konfigurasi terbaiknya dari ablasi (02), lalu
GridSearchCV membungkus seluruh pipeline (vektorisasi + ROS + classifier).
Data uji hanya dipakai sekali di sini.

Output:
  outputs/tables/grid_search.csv                parameter terpilih        [ISI-8]
  outputs/tables/model_akhir.csv                metrik uji + waktu latih  [ISI-4]
  outputs/tables/classification_report_*.csv                              [ISI-7]
  outputs/tables/mcnemar.csv                    nilai p tiap pasangan     [ISI-6]
  outputs/tables/lr_top20_kata.csv              koefisien LR per kelas    [ISI-7]
  outputs/figures/confusion_matrix_*.png, lr_top20_kata.png               [ISI-7]
  outputs/models/*.joblib, final_config.json
"""
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import GridSearchCV
from statsmodels.stats.contingency_tables import mcnemar

from common import (ALGO_NAMES, BEST_JSON, CLASSES, FIGURES, MODELS, PARAM_GRID, ROOT, TABLES,
                    TEXT_COLS, ensure_dirs, load_clean, make_cv, make_pipeline, save_json,
                    save_table, split)

N_TIMING_RUNS = 5


def best_configs():
    path = TABLES / "ablasi_24_konfigurasi.csv"
    if not path.exists():
        raise SystemExit(f"{path} belum ada. Jalankan src/02_ablation.py lebih dulu.")
    abl = pd.read_csv(path)
    return {algo: abl[abl["algoritma"] == algo].sort_values("f1_macro_mean", ascending=False).iloc[0]
            for algo in ["nb", "svm", "lr"]}


def mcnemar_row(name_a, name_b, pred_a, pred_b, y):
    a_ok, b_ok = (pred_a == y), (pred_b == y)
    tabel = [[np.sum(a_ok & b_ok), np.sum(a_ok & ~b_ok)],
             [np.sum(~a_ok & b_ok), np.sum(~a_ok & ~b_ok)]]
    n_disc = tabel[0][1] + tabel[1][0]
    # Uji eksak (binomial) bila pasangan yang berbeda sedikit; chi-square dengan koreksi kontinuitas bila banyak
    exact = n_disc < 25
    if n_disc == 0:
        stat, p = float("nan"), 1.0
    else:
        res = mcnemar(tabel, exact=exact, correction=True)
        stat, p = res.statistic, res.pvalue
    return {"model_a": name_a, "model_b": name_b,
            "a_benar_b_salah": int(tabel[0][1]), "a_salah_b_benar": int(tabel[1][0]),
            "uji": "eksak" if exact else "chi-square", "statistik": stat, "p_value": p,
            "signifikan_0.05": bool(p < 0.05)}


def top_words(pipe, n=20):
    vocab = np.array(pipe.named_steps["vec"].get_feature_names_out())
    clf = pipe.named_steps["clf"]
    rows = []
    for k, cls in enumerate(clf.classes_):
        coef = clf.coef_[k]
        for rank, j in enumerate(np.argsort(coef)[::-1][:n], 1):
            rows.append({"kelas": cls, "peringkat": rank, "kata": vocab[j], "koefisien": coef[j]})
    return pd.DataFrame(rows)


def main():
    ensure_dirs()
    df = load_clean()
    train, test = split(df)
    y_train, y_test = train["rating_class"].values, test["rating_class"].values
    configs = best_configs()

    grid_rows, final_rows, preds, pipes, params = [], [], {}, {}, {}
    for algo, cfg in configs.items():
        text_col = TEXT_COLS[(bool(cfg["lemmatization"]), True)]
        pipe = make_pipeline(algo, cfg["vektorisasi"], bool(cfg["ros"]))
        gs = GridSearchCV(pipe, PARAM_GRID[algo], cv=make_cv(), scoring="f1_macro", n_jobs=-1)
        gs.fit(train[text_col], y_train)
        (param, values), = PARAM_GRID[algo].items()
        for v, m, s in zip(gs.cv_results_[f"param_{param}"], gs.cv_results_["mean_test_score"],
                           gs.cv_results_["std_test_score"]):
            grid_rows.append({"algoritma": ALGO_NAMES[algo], "parameter": param.replace("clf__", ""),
                              "nilai": v, "f1_macro_cv_mean": m, "f1_macro_cv_std": s,
                              "terpilih": v == gs.best_params_[param]})

        # Waktu latih diukur ulang pada model akhir agar tidak terpengaruh paralelisme CV
        final = make_pipeline(algo, cfg["vektorisasi"], bool(cfg["ros"]))
        final.set_params(**gs.best_params_)
        # Median dari beberapa kali latih, karena satu pengukuran waktu terlalu bising
        times = []
        for _ in range(N_TIMING_RUNS):
            t0 = time.perf_counter()
            final.fit(train[text_col], y_train)
            times.append(time.perf_counter() - t0)
        fit_time = float(np.median(times))
        t0 = time.perf_counter()
        pred = final.predict(test[text_col])
        pred_time = time.perf_counter() - t0

        preds[algo], pipes[algo] = pred, final
        params[algo] = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
        joblib.dump(final, MODELS / f"model_{algo}.joblib")
        final_rows.append({
            "algoritma": ALGO_NAMES[algo], "kode": algo, "vektorisasi": cfg["vektorisasi"],
            "lemmatization": bool(cfg["lemmatization"]), "ros": bool(cfg["ros"]),
            "parameter_terpilih": str(params[algo]),
            "f1_macro_cv": gs.best_score_,
            "accuracy_uji": accuracy_score(y_test, pred),
            "balanced_accuracy_uji": balanced_accuracy_score(y_test, pred),
            "f1_macro_uji": f1_score(y_test, pred, average="macro"),
            "waktu_latih_detik_median": fit_time, "waktu_prediksi_detik": pred_time,
        })

        rep = pd.DataFrame(classification_report(y_test, pred, labels=CLASSES, output_dict=True)).T
        save_table(rep.round(4), f"classification_report_{algo}.csv", index=True)
        cm = confusion_matrix(y_test, pred, labels=CLASSES)
        save_table(pd.DataFrame(cm, index=[f"aktual_{c}" for c in CLASSES],
                                columns=[f"prediksi_{c}" for c in CLASSES]),
                   f"confusion_matrix_{algo}.csv", index=True)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Confusion Matrix: {ALGO_NAMES[algo]}")
        fig.tight_layout()
        fig.savefig(FIGURES / f"confusion_matrix_{algo}.png", dpi=150)
        plt.close(fig)
        print(f"{ALGO_NAMES[algo]}: {gs.best_params_}  f1_uji={final_rows[-1]['f1_macro_uji']:.4f}")

    save_table(pd.DataFrame(grid_rows).round(4), "grid_search.csv")
    final_df = pd.DataFrame(final_rows)
    save_table(final_df.round(4), "model_akhir.csv")

    # Model terbaik dipilih dari skor CV, bukan dari data uji
    best_algo = final_df.sort_values("f1_macro_cv", ascending=False).iloc[0]["kode"]
    others = [a for a in ["nb", "svm", "lr"] if a != best_algo]
    pairs = [(best_algo, o) for o in others]
    if best_algo != "lr":
        pairs += [("lr", o) for o in others if o != "lr"]
    mc = pd.DataFrame([mcnemar_row(ALGO_NAMES[a], ALGO_NAMES[b], preds[a], preds[b], y_test)
                       for a, b in pairs])
    save_table(mc.round(4), "mcnemar.csv")
    print(mc.round(4).to_string(index=False))

    tw = top_words(pipes["lr"])
    save_table(tw.round(4), "lr_top20_kata.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax, cls in zip(axes, CLASSES):
        d = tw[tw["kelas"] == cls].iloc[::-1]
        ax.barh(d["kata"], d["koefisien"], color="#4C72B0")
        ax.set_title(f"Kelas {cls}")
        ax.set_xlabel("Koefisien Logistic Regression")
    fig.tight_layout()
    fig.savefig(FIGURES / "lr_top20_kata.png", dpi=150)
    plt.close(fig)

    cfg = final_df.set_index("kode").loc[best_algo]
    save_json({
        "best_algo_cv": best_algo,
        "configs": {r["kode"]: {"vektorisasi": r["vektorisasi"], "lemmatization": bool(r["lemmatization"]),
                                "ros": bool(r["ros"]),
                                "params": params[r["kode"]]}
                    for r in final_rows},
        "catatan": "Model terbaik dipilih dari f1_macro CV; lihat mcnemar.csv untuk signifikansi.",
    }, BEST_JSON)
    print(f"Model terbaik (CV): {ALGO_NAMES[best_algo]}, f1_uji={cfg['f1_macro_uji']:.4f}")


if __name__ == "__main__":
    main()
