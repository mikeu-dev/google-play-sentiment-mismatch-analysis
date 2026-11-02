"""Lima gambar untuk Subbab 4.4.3 tanpa Tableau (pengganti/cadangan).

Membaca outputs/tableau/tableau_reviews.csv (hasil 08), sehingga tidak perlu
menjalankan ulang 01-04. Gambar disimpan ke outputs/figures/ dan antrean
prioritas lengkap ke outputs/tables/antrean_prioritas.csv.

Gambar 4.1 ringkasan, 4.2 peta rating terhadap prediksi, 4.3 antrean prioritas,
4.4 keluhan tersembunyi per aplikasi, 4.5 panjang ulasan terhadap keyakinan.
"""
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "outputs" / "tableau" / "tableau_reviews.csv"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"

INK, MUTED, GRID = "#1f2328", "#59636e", "#d1d9e0"
GREY, ACCENT = "#b6bec8", "#1f6feb"
TIER_ALPHA = {"tinggi": 1.0, "sedang": 0.55, "rendah": 0.25}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})

LABEL = {"sesuai": "Sesuai", "keluhan_tersembunyi": "Keluhan tersembunyi (bintang 4–5 → Negative)",
         "pujian_tersembunyi": "Pujian tersembunyi (bintang 1–2 → Positive)",
         "rating3_tidak_netral": "Rating 3 tidak netral", "lainnya": "Lainnya (→ Neutral)"}


def idn(x, d=0):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def save(fig, name):
    path = FIG / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path.relative_to(ROOT)}")


def clean(text, n=110):
    text = re.sub(r"\S+@\S+|https?://\S+|\+?\d[\d\s-]{7,}\d", "[dihapus]", str(text))
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def fig1(d):
    order = ["sesuai", "lainnya", "rating3_tidak_netral", "pujian_tersembunyi", "keluhan_tersembunyi"]
    n = d["mismatch_type"].value_counts().reindex(order)
    tier = pd.crosstab(d["mismatch_type"], d["confidence_tier"]).reindex(order)[["tinggi", "sedang", "rendah"]]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for i, k in enumerate(order):
        if k == "sesuai":
            ax.barh(i, n[k], color=GREY, height=0.55)
        else:
            left = 0
            for t in ["tinggi", "sedang", "rendah"]:
                ax.barh(i, tier.loc[k, t], left=left, height=0.55, color=ACCENT, alpha=TIER_ALPHA[t],
                        edgecolor="white", linewidth=1)
                left += tier.loc[k, t]
        extra = f" ({idn(tier.loc[k, 'tinggi'])} keyakinan tinggi)" if k != "sesuai" else ""
        ax.text(n[k] + 80, i, f"{idn(n[k])} · {idn(n[k] / len(d) * 100, 1)}%{extra}", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(order)), [LABEL[k] for k in order], color=INK)
    ax.set_xlim(0, n.max() * 1.45)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    hi = d[(d["score"] >= 4)]
    k409 = int((d["mismatch_type"] == "keluhan_tersembunyi").sum())
    k126 = int(d["is_priority"].sum())
    fig.suptitle(f"{idn((1 - d['is_mismatch'].mean()) * 100, 2)}% ulasan sesuai; {k409} kandidat keluhan tersembunyi, "
                 f"hanya {k126} berkeyakinan sedang atau tinggi", x=0.02, ha="left", fontsize=12.5, color=INK, y=1.0)
    ax.set_title(f"Jumlah ulasan per tipe ketidaksesuaian (n = {idn(len(d))}; prediksi out-of-fold Logistic Regression)",
                 loc="left", fontsize=9.5, color=MUTED)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ACCENT, alpha=a) for a in TIER_ALPHA.values()]
    ax.legend(handles, ["keyakinan tinggi (≥ 0,80)", "sedang (0,60–0,79)", "rendah (< 0,60)"],
              loc="lower right", frameon=False, fontsize=8.5, labelcolor=MUTED)
    save(fig, "gambar_4_1_ringkasan.png")


def fig2(d):
    ct = pd.crosstab(d["score"], d["pred"]).reindex(columns=["negative", "neutral", "positive"])
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.imshow(pct.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(5):
        for j in range(3):
            v = pct.values[i, j]
            ax.text(j, i, f"{idn(v, 1)}%\n({idn(ct.values[i, j])})", ha="center", va="center", fontsize=9,
                    color="white" if v > 55 else INK)
    ax.set_xticks(range(3), ["Negative", "Neutral", "Positive"])
    ax.set_yticks(range(5), [f"{s} bintang" for s in ct.index])
    ax.set_xlabel("Prediksi teks (out-of-fold)")
    ax.set_ylabel("Rating pengguna")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.suptitle("Ulasan bintang 4–5 yang diprediksi Negative: 11,0% (bintang 4) dan 6,1% (bintang 5)",
                 x=0.02, ha="left", fontsize=11.5, color=INK, y=1.0)
    ax.set_title("Persentase per baris rating (jumlah ulasan dalam kurung)", loc="left", fontsize=9.5, color=MUTED)
    save(fig, "gambar_4_2_peta_rating_prediksi.png")


def fig3(d):
    q = d[d["is_priority"]].sort_values("proba_neg", ascending=False)
    out = q[["review_id", "app_id", "score", "proba_neg", "confidence_tier", "review_date", "content"]].copy()
    out["content"] = out["content"].map(lambda t: clean(t, 400))
    out.to_csv(TAB / "antrean_prioritas.csv", index=False)
    print(f"  -> outputs/tables/antrean_prioritas.csv ({len(out)} baris)")
    top = q.head(10)
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.axis("off")
    cols = ["No", "Bintang", "Aplikasi", "P(Negative)", "Cuplikan ulasan"]
    rows = [[str(i + 1), str(int(r.score)), r.app_id.split(".")[-1], idn(r.proba_neg, 2),
             "\n".join(textwrap.wrap(clean(r.content, 150), 62))] for i, r in enumerate(top.itertuples())]
    t = ax.table(cellText=rows, colLabels=cols, loc="upper center", cellLoc="left",
                 colWidths=[0.05, 0.08, 0.14, 0.11, 0.62])
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1, 2.35)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.6)
        cell.PAD = 0.02
        if r == 0:
            cell.set_facecolor("#eef2f6")
            cell.set_text_props(color=INK, weight="bold")
        else:
            cell.set_text_props(color=INK)
    fig.suptitle(f"Antrean prioritas: 10 dari {len(q)} keluhan tersembunyi berkeyakinan sedang atau tinggi",
                 x=0.02, ha="left", fontsize=12, color=INK, y=0.99)
    fig.text(0.02, 0.93, "Ulasan bintang 4–5 yang diprediksi Negative, diurutkan menurut probabilitas Negative. "
             "Daftar lengkap: outputs/tables/antrean_prioritas.csv", fontsize=8.5, color=MUTED)
    save(fig, "gambar_4_3_antrean_prioritas.png")


def fig4(d):
    hi = d[d["score"] >= 4]
    g = hi.groupby("app_id").agg(n=("score", "size"), k=("mismatch_type", lambda s: (s == "keluhan_tersembunyi").sum()))
    g["rate"] = g["k"] / g["n"] * 100
    g = g.sort_values("rate")
    overall = hi["mismatch_type"].eq("keluhan_tersembunyi").mean() * 100
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    y = np.arange(len(g))
    ax.barh(y, g["rate"], color=[ACCENT if r > overall else GREY for r in g["rate"]], height=0.6)
    for yi, (idx, r) in zip(y, g.iterrows()):
        ax.text(r["rate"] + 0.25, yi, f"{idn(r['rate'], 1)}% ({int(r['k'])} dari {int(r['n'])})", va="center", fontsize=8.5, color=INK)
    ax.axvline(overall, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(overall, len(g) - 0.1, f" rata-rata {idn(overall, 1)}%", color=MUTED, fontsize=8.5, va="bottom")
    ax.set_yticks(y, g.index, color=INK, fontsize=8.5)
    ax.set_xlim(0, g["rate"].max() * 1.35)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    fig.suptitle(f"Tingkat kandidat keluhan tersembunyi berkisar {idn(g['rate'].min(), 1)}% sampai {idn(g['rate'].max(), 1)}% antaraplikasi",
                 x=0.02, ha="left", fontsize=12, color=INK, y=1.0)
    ax.set_title("Persentase ulasan bintang 4–5 yang teksnya diprediksi Negative; biru = di atas rata-rata", loc="left", fontsize=9.5, color=MUTED)
    save(fig, "gambar_4_4_keluhan_per_aplikasi.png")


def fig5(d):
    rho = d["n_char"].corr(d["confidence"], method="spearman")
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    s = d.sample(min(len(d), 6000), random_state=42)
    for flag, col, lab, z in [(False, GREY, "sesuai", 1), (True, ACCENT, "mismatch", 2)]:
        m = s[s["is_mismatch"] == flag]
        ax.scatter(m["n_char"], m["confidence"], s=7, color=col, alpha=0.35 if not flag else 0.5, label=lab, linewidths=0, zorder=z)
    xs = d["n_char"].clip(lower=1)
    bins = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 12)
    med = d.groupby(pd.cut(xs, bins), observed=True)["confidence"].median()
    ctr = [np.sqrt(iv.left * iv.right) for iv in med.index]
    ax.plot(ctr, med.values, color=INK, lw=1.8, zorder=3, label="median per rentang panjang")
    ax.set_xscale("log")
    ax.set_xlabel("Panjang ulasan (karakter, skala log)")
    ax.set_ylabel("Keyakinan model (probabilitas kelas terpilih)")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=MUTED, loc="lower left", markerscale=2)
    fig.suptitle(f"Ulasan lebih panjang sedikit kurang diyakini model (Spearman ρ = {idn(rho, 3)})",
                 x=0.02, ha="left", fontsize=12, color=INK, y=1.0)
    ax.set_title(f"Sampel acak 6.000 dari {idn(len(d))} ulasan; garis = median keyakinan", loc="left", fontsize=9.5, color=MUTED)
    save(fig, "gambar_4_5_panjang_keyakinan.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(CSV)
    fig1(d); fig2(d); fig3(d); fig4(d); fig5(d)


if __name__ == "__main__":
    main()
