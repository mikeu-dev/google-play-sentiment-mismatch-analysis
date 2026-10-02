"""Pembersih teks (Tahap 2).

Perbedaan dengan versi notebook:
1. Kata negasi (no, not, nor, never, dan bentuk n't) dipertahankan, karena
   daftar stopword NLTK memuatnya dan menghapusnya membalik makna ulasan.
2. Lemmatization memakai POS tag; tanpa itu WordNetLemmatizer menganggap
   semua kata sebagai kata benda ("crashed" tidak kembali ke "crash").
3. Karakter non-huruf diganti spasi, bukan dihapus, agar "app.great" tidak
   menjadi satu token "appgreat".
"""
import re

import nltk
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

NLTK_PACKAGES = {
    "corpora/stopwords": "stopwords",
    "corpora/wordnet": "wordnet",
    "corpora/omw-1.4": "omw-1.4",
    "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
}


def ensure_nltk():
    for resource, package in NLTK_PACKAGES.items():
        for name in (resource, resource + ".zip"):
            try:
                nltk.data.find(name)
                break
            except LookupError:
                continue
        else:
            nltk.download(package, quiet=True)


ensure_nltk()

NEGASI = {"no", "not", "nor", "never"}
STOP_ALL = set(stopwords.words("english"))
STOP_KEEP_NEG = STOP_ALL - NEGASI

_lemmatizer = WordNetLemmatizer()
_POS = {"J": wordnet.ADJ, "V": wordnet.VERB, "R": wordnet.ADV}


def normalisasi_negasi(text):
    text = str(text).lower().replace("’", "'")
    text = text.replace("can't", "can not").replace("won't", "will not")
    return re.sub(r"n't\b", " not", text)   # harus sebelum regex [^a-z\s]


def tokenize(text):
    text = normalisasi_negasi(text)
    return re.sub(r"[^a-z\s]", " ", text).split()


def lemmatize(tokens):
    # POS tag diberikan pada kalimat utuh, sebelum stopword dibuang
    return [_lemmatizer.lemmatize(w, _POS.get(t[0], wordnet.NOUN)) for w, t in pos_tag(tokens)]


def clean_text(text, lemma=True, keep_negation=True):
    tokens = tokenize(text)
    if lemma and tokens:
        tokens = lemmatize(tokens)
    stop = STOP_KEEP_NEG if keep_negation else STOP_ALL
    return " ".join(w for w in tokens if w not in stop and len(w) > 1)
