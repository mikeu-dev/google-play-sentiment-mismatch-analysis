"""Konstanta, path, dan fungsi bersama untuk seluruh skrip 01-08.

Setiap angka di Bab IV berasal dari salah satu skrip di folder ini dan
tersimpan di ``outputs/``. Semua sumber keacakan memakai ``SEED`` yang sama.
"""
import json
import os
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

SEED = 42
TEST_SIZE = 0.2
N_FOLDS = 5
CLASSES = ["negative", "neutral", "positive"]   # urutan kolom predict_proba

# Ambang tingkat keyakinan (Subbab 3.4.2)
TIER_HIGH = 0.80
TIER_MED = 0.60

# Rentang Grid Search (Subbab 3.2.5)
PARAM_GRID = {
    "nb": {"clf__alpha": [0.1, 0.5, 1.0]},
    "svm": {"clf__C": [0.1, 1, 10]},
    "lr": {"clf__C": [0.1, 1, 10]},
}
ALGO_NAMES = {"nb": "Multinomial Naive Bayes", "svm": "Linear SVM", "lr": "Logistic Regression"}

# Jumlah sampel validasi manual (Subbab 3.4.3)
SAMPLE_SIZES = {
    "keluhan_tersembunyi": 150,
    "pujian_tersembunyi": 75,
    "rating3_tidak_netral": 75,
    "kontrol": 100,
}
N_SECOND_ANNOTATOR = 100

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = Path(os.environ.get("REVIEWS_CSV", ROOT / "data" / "raw" / "reviews.csv"))
PROCESSED = ROOT / "data" / "processed"
ANNOTATION = ROOT / "data" / "annotation"
OUT = ROOT / "outputs"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
TABLEAU = OUT / "tableau"
MODELS = OUT / "models"

CLEAN_CSV = PROCESSED / "reviews_clean.csv"
OOF_CSV = PROCESSED / "oof_predictions.csv"
VALIDATED_CSV = PROCESSED / "validated_labels.csv"
BEST_JSON = MODELS / "final_config.json"

# Varian teks bersih hasil 01_preprocess.py; (lemma, keep_negation) -> kolom
TEXT_COLS = {
    (True, True): "text_lemma_neg",
    (False, True): "text_nolemma_neg",
    (True, False): "text_lemma_noneg",
    (False, False): "text_nolemma_noneg",
}
MAIN_TEXT = TEXT_COLS[(True, True)]


def ensure_dirs():
    for d in (PROCESSED, ANNOTATION, TABLES, FIGURES, TABLEAU, MODELS):
        d.mkdir(parents=True, exist_ok=True)


def rating_class(score):
    return "positive" if score >= 4 else "neutral" if score == 3 else "negative"


def load_clean():
    if not CLEAN_CSV.exists():
        raise SystemExit(f"{CLEAN_CSV} belum ada. Jalankan src/01_preprocess.py lebih dulu.")
    df = pd.read_csv(CLEAN_CSV, keep_default_na=False, na_values=[""])
    for col in TEXT_COLS.values():
        df[col] = df[col].fillna("")
    return df


def load_oof():
    if not OOF_CSV.exists():
        raise SystemExit(f"{OOF_CSV} belum ada. Jalankan src/04_oof_mismatch.py lebih dulu.")
    return pd.read_csv(OOF_CSV)


def make_vectorizer(kind):
    if kind == "tfidf":
        return TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    if kind == "count":
        return CountVectorizer(max_features=5000, ngram_range=(1, 2))
    raise ValueError(kind)


def make_classifier(algo, calibrate=False):
    if algo == "nb":
        return MultinomialNB()
    if algo == "lr":
        return LogisticRegression(max_iter=2000, random_state=SEED)
    if algo == "svm":
        svm = LinearSVC(dual=False, random_state=SEED)
        # LinearSVC tidak punya predict_proba; kalibrasi hanya dipakai bila
        # probabilitas dibutuhkan (Tahap 4).
        return CalibratedClassifierCV(svm, cv=3) if calibrate else svm
    raise ValueError(algo)


def make_pipeline(algo, vectorizer="tfidf", use_ros=True, calibrate=False):
    """TF-IDF dan ROS di dalam satu pipeline agar hanya belajar dari data latih.

    ``imblearn.pipeline.Pipeline`` menjalankan ROS hanya saat ``fit``,
    tidak saat ``predict``, sehingga data validasi/uji tidak pernah diduplikasi.
    """
    steps = [("vec", make_vectorizer(vectorizer))]
    if use_ros:
        steps.append(("ros", RandomOverSampler(random_state=SEED)))
    steps.append(("clf", make_classifier(algo, calibrate)))
    return Pipeline(steps)


def make_cv():
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def split(df):
    """Pembagian latih/uji yang sama untuk semua skrip (kolom ``split`` dari 01)."""
    return df[df["split"] == "train"], df[df["split"] == "test"]


def save_table(df, name, index=False):
    path = TABLES / name
    df.to_csv(path, index=index)
    print(f"  -> {path.relative_to(ROOT)}")
    return path


def save_json(obj, path):
    path = Path(path)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
    print(f"  -> {path.relative_to(ROOT)}")


def load_json(path):
    return json.loads(Path(path).read_text())
