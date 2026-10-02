"""Tahap 1-2: muat data, bersihkan teks, hapus duplikat, dan bagi latih/uji.

Output:
  data/processed/reviews_clean.csv     satu baris per ulasan + kolom ``split``
  outputs/tables/kolom_dataset.csv      daftar kolom dataset mentah (cek replyContent)
  outputs/tables/data_cleaning.csv      jumlah ulasan dikeluarkan per langkah  [ISI-3]
  outputs/tables/distribusi_kelas.csv   Tabel 4.1                              [ISI-3]

Pembersihan teks tidak belajar dari data, sehingga aman dijalankan sekali
sebelum pembagian. TF-IDF dan ROS baru dijalankan di dalam pipeline (02-04).
"""
import pandas as pd
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split

from common import (CLEAN_CSV, MAIN_TEXT, RAW_CSV, ROOT, SEED, TEST_SIZE, TEXT_COLS,
                    ensure_dirs, rating_class, save_table)
from preprocessing import NEGASI, STOP_ALL, STOP_KEEP_NEG, _POS, _lemmatizer, pos_tag, tokenize

# Kolom metadata yang ikut disimpan (nama asli -> nama baru). Nama dan foto
# pengguna sengaja tidak disimpan.
META_COLS = {
    "reviewId": "review_id",
    "appId": "app_id",
    "at": "review_date",
    "reviewCreatedVersion": "app_version",
    "thumbsUpCount": "thumbs_up",
    "replyContent": "replyContent",
    "repliedAt": "repliedAt",
    "sortOrder": "sort_order",
}


def load_raw():
    if RAW_CSV.exists():
        return pd.read_csv(RAW_CSV)
    try:
        import kagglehub
    except ImportError:
        raise SystemExit(f"{RAW_CSV} tidak ditemukan dan kagglehub tidak terpasang.")
    path = kagglehub.dataset_download("prakharrathi25/google-play-store-reviews")
    df = pd.read_csv(f"{path}/reviews.csv")
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_CSV, index=False)
    return df


def clean_variants(text):
    """Empat varian teks bersih sekaligus; POS tag hanya dihitung sekali."""
    tokens = tokenize(text)
    lemmas = [_lemmatizer.lemmatize(w, _POS.get(t[0], wordnet.NOUN))
              for w, t in pos_tag(tokens)] if tokens else []
    out = {}
    for (lemma, keep_neg), col in TEXT_COLS.items():
        stop = STOP_KEEP_NEG if keep_neg else STOP_ALL
        src = lemmas if lemma else tokens
        out[col] = " ".join(w for w in src if w not in stop and len(w) > 1)
    return out


def main():
    ensure_dirs()
    raw = load_raw()
    print(f"Data mentah: {len(raw)} baris, kolom: {list(raw.columns)}")
    save_table(pd.DataFrame({"kolom": raw.columns,
                             "terisi": [raw[c].notna().sum() for c in raw.columns]}),
               "kolom_dataset.csv")

    log = [("Data mentah", 0, len(raw))]

    def step(name, before, after):
        log.append((name, before - after, after))
        print(f"{name}: -{before - after} -> {after}")

    df = raw.rename(columns=META_COLS)
    keep = ["content", "score"] + [c for c in META_COLS.values() if c in df.columns]
    df = df[keep].copy()

    n = len(df)
    df = df[df["content"].notna() & (df["content"].astype(str).str.strip() != "")]
    step("Konten kosong (NaN)", n, len(df))

    if "review_id" in df.columns:
        # Dataset di-scrape dengan beberapa sortOrder; ulasan yang sama bisa muncul dua kali.
        n = len(df)
        df = df.drop_duplicates(subset="review_id")
        step("Duplikat reviewId", n, len(df))
    else:
        df["review_id"] = [f"r{i:05d}" for i in range(len(df))]

    df["rating_class"] = df["score"].apply(rating_class)
    df["n_char"] = df["content"].astype(str).str.len()

    print("Membersihkan teks (POS tag + lemmatization)...")
    variants = pd.DataFrame([clean_variants(t) for t in df["content"]], index=df.index)
    df = pd.concat([df, variants], axis=1)
    df["n_token"] = df[MAIN_TEXT].str.split().str.len().fillna(0).astype(int)

    n = len(df)
    df = df[df["n_token"] > 0]
    step("Teks kosong setelah dibersihkan (n_token == 0)", n, len(df))
    n_short = int((df["n_token"] < 3).sum())

    dup = df.groupby(MAIN_TEXT)["rating_class"].agg(["size", "nunique"])
    dup = dup[dup["size"] > 1]
    n = len(df)
    df = df.drop_duplicates(subset=MAIN_TEXT, keep="first")
    step("Duplikat teks bersih (identik)", n, len(df))

    train_idx, test_idx = train_test_split(df.index, test_size=TEST_SIZE, random_state=SEED,
                                           stratify=df["rating_class"])
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"

    log_df = pd.DataFrame(log, columns=["langkah", "dikeluarkan", "sisa"])
    log_df.loc[len(log_df)] = ["(info) ulasan dengan < 3 token, tetap dipakai", n_short, len(df)]
    log_df.loc[len(log_df)] = ["(info) kelompok teks duplikat", len(dup), None]
    log_df.loc[len(log_df)] = ["(info) kelompok duplikat dengan rating_class berbeda",
                               int((dup["nunique"] > 1).sum()), None]
    log_df.loc[len(log_df)] = ["Data latih", None, len(train_idx)]
    log_df.loc[len(log_df)] = ["Data uji", None, len(test_idx)]
    save_table(log_df, "data_cleaning.csv")
    save_table(dup.sort_values("size", ascending=False).head(30).reset_index()
               .rename(columns={MAIN_TEXT: "teks", "size": "jumlah", "nunique": "kelas_berbeda"}),
               "duplikat_teratas.csv")

    dist = pd.crosstab(df["rating_class"], df["split"], margins=True, margins_name="total")
    dist["persen"] = (dist["total"] / dist.loc["total", "total"] * 100).round(2)
    save_table(dist.reset_index(), "distribusi_kelas.csv")
    save_table(df.groupby("score").size().rename("jumlah").reset_index(), "distribusi_rating.csv")

    df.to_csv(CLEAN_CSV, index=False)
    print(f"  -> {CLEAN_CSV.relative_to(ROOT)} ({len(df)} baris)")


if __name__ == "__main__":
    main()
