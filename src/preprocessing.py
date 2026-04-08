"""
preprocessing.py
================
Pipeline preprocessing teks Bahasa Indonesia (informal/slang)
untuk keperluan sentiment classification.

Penggunaan cepat:
    from preprocessing import TextPreprocessor, TFIDFVectorizer, PreprocessingPipeline

    pipeline = PreprocessingPipeline()
    X_df     = pipeline.fit_transform("dataset.csv")          # fit + transform
    X_new    = pipeline.transform_raw(["barang bagus bgt!"])   # prediksi baru
    pipeline.save("pipeline.pkl")                             # simpan
    pipeline.load("pipeline.pkl")                             # muat ulang
"""

from __future__ import annotations

import re
import pickle
import logging
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as _SklearnTFIDF

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 0. _PreprocessingUnpickler
#    Custom Unpickler untuk menangani kasus pipeline.pkl yang disimpan saat
#    preprocessing.py dijalankan langsung (__main__), kemudian dimuat dari
#    file/modul lain (misal klasifikasi.py).
# ══════════════════════════════════════════════════════════════════════════════
class _PreprocessingUnpickler(pickle.Unpickler):
    """
    Custom Unpickler yang me-remap referensi '__main__' ke modul 'preprocessing'.
    Digunakan secara internal oleh PreprocessingPipeline.load().
    """

    # Nama-nama class yang didefinisikan di file ini
    _LOCAL_CLASSES = {"TextPreprocessor", "TFIDFVectorizer", "PreprocessingPipeline"}

    def find_class(self, module: str, name: str):
        import importlib

        if module == "__main__" and name in self._LOCAL_CLASSES:
            try:
                mod = importlib.import_module("preprocessing")
                return getattr(mod, name)
            except (ImportError, AttributeError):
                current_module = importlib.import_module(__name__)
                return getattr(current_module, name)

        return super().find_class(module, name)


# ══════════════════════════════════════════════════════════════════════════════
# 1. TextPreprocessor
#    Bertanggung jawab membersihkan raw teks menjadi teks bersih.
# ══════════════════════════════════════════════════════════════════════════════
class TextPreprocessor:
    """
    Membersihkan teks Bahasa Indonesia informal/slang.

    Tahapan (dapat dikonfigurasi):
        1. Lowercase
        2. Hapus URL
        3. Hapus mention & hashtag
        4. Hapus tanda baca
        5. Hapus angka
        6. Normalisasi slang (jika slang_dict diberikan)
        7. Normalisasi whitespace

    Parameters
    ----------
    lowercase : bool, default True
        Ubah semua karakter ke huruf kecil.
    remove_url : bool, default True
        Hapus URL (http/https/www).
    remove_mention_hashtag : bool, default True
        Hapus @mention dan #hashtag.
    remove_punctuation : bool, default True
        Hapus semua tanda baca.
    remove_numbers : bool, default True
        Hapus semua angka.
    min_token_length : int, default 2
        Hapus token yang panjangnya kurang dari nilai ini.
    slang_dict : dict or str/Path, optional
        Dictionary mapping {slang: formal} atau path ke file CSV yang berisi
        kolom 'slang' dan 'formal'.
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_url: bool = True,
        remove_mention_hashtag: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = True,
        min_token_length: int = 2,
        slang_dict: Optional[Union[dict, str, Path]] = None,
    ) -> None:
        self.lowercase = lowercase
        self.remove_url = remove_url
        self.remove_mention_hashtag = remove_mention_hashtag
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.min_token_length = min_token_length

        self._url_re = re.compile(r"https?://\S+|www\.\S+")
        self._mention_re = re.compile(r"@\w+")
        self._hashtag_re = re.compile(r"#\w+")
        self._punct_re = re.compile(r"[^\w\s]")
        self._number_re = re.compile(r"\d+")
        self._whitespace_re = re.compile(r"\s+")
        
        self.slang_mapping = {}
        if isinstance(slang_dict, dict):
            self.slang_mapping = slang_dict
        elif slang_dict is not None:
            self._load_slang_csv(slang_dict)

    def _load_slang_csv(self, path: Union[str, Path]) -> None:
        """Load slang mapping dari file CSV."""
        path = Path(path)
        if not path.exists():
            logger.warning("File slang tidak ditemukan: %s", path.resolve())
            return
            
        try:
            df = pd.read_csv(path)
            if 'slang' in df.columns and 'formal' in df.columns:
                df = df.dropna(subset=['formal'])
                self.slang_mapping = dict(zip(df['slang'], df['formal']))
                logger.info("Berhasil memuat %d kata slang dari %s", len(self.slang_mapping), path.name)
            else:
                logger.warning("File CSV slang harus memiliki kolom 'slang' dan 'formal'.")
        except Exception as e:
            logger.error("Gagal memuat file slang: %s", e)

    def clean(self, text: str) -> str:
        """Bersihkan satu string teks."""
        if not isinstance(text, str):
            text = str(text)

        if self.lowercase:
            text = text.lower()
        if self.remove_url:
            text = self._url_re.sub(" ", text)
        if self.remove_mention_hashtag:
            text = self._mention_re.sub(" ", text)
            text = self._hashtag_re.sub(" ", text)
        if self.remove_punctuation:
            text = self._punct_re.sub(" ", text)
        if self.remove_numbers:
            text = self._number_re.sub("", text)

        tokens = self._whitespace_re.split(text.strip())
        
        cleaned_tokens = []
        for t in tokens:
            if len(t) >= self.min_token_length:
                normalized = self.slang_mapping.get(t, t)
                cleaned_tokens.append(str(normalized))
                
        return " ".join(cleaned_tokens)

    def transform(self, texts: Union[List[str], pd.Series]) -> List[str]:
        """Bersihkan list/Series teks sekaligus."""
        logger.info("TextPreprocessor: membersihkan %d teks...", len(texts))
        cleaned = [self.clean(t) for t in texts]
        logger.info("TextPreprocessor: selesai.")
        return cleaned

    def __repr__(self) -> str:
        return (
            f"TextPreprocessor("
            f"lowercase={self.lowercase}, "
            f"remove_url={self.remove_url}, "
            f"remove_punctuation={self.remove_punctuation}, "
            f"min_token_length={self.min_token_length})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2. TFIDFVectorizer
#    Wrapper sklearn TfidfVectorizer dengan default optimal untuk
#    dataset teks pendek Bahasa Indonesia.
# ══════════════════════════════════════════════════════════════════════════════
class TFIDFVectorizer:
    """
    Wrapper TF-IDF dengan default yang dioptimalkan untuk teks
    Bahasa Indonesia informal (pendek, banyak slang).
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple = (1, 2),
        min_df: int = 3,
        max_df: float = 0.90,
        sublinear_tf: bool = True,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self._is_fitted = False

        self._vectorizer = _SklearnTFIDF(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            analyzer="word",
        )

    def fit(self, texts: Union[List[str], pd.Series]) -> "TFIDFVectorizer":
        """Fit vectorizer pada kumpulan teks."""
        logger.info("TFIDFVectorizer: fitting pada %d teks...", len(texts))
        self._vectorizer.fit(texts)
        self._is_fitted = True
        logger.info(
            "TFIDFVectorizer: selesai. Jumlah fitur = %d",
            len(self._vectorizer.get_feature_names_out()),
        )
        return self

    def transform(self, texts: Union[List[str], pd.Series]) -> pd.DataFrame:
        """Transform teks menjadi DataFrame TF-IDF."""
        self._check_fitted()
        logger.info("TFIDFVectorizer: transforming %d teks...", len(texts))
        matrix = self._vectorizer.transform(texts)
        df = pd.DataFrame(
            matrix.toarray(),
            columns=self._vectorizer.get_feature_names_out(),
        )
        logger.info("TFIDFVectorizer: output shape = %s", df.shape)
        return df

    def fit_transform(self, texts: Union[List[str], pd.Series]) -> pd.DataFrame:
        """Fit sekaligus transform."""
        return self.fit(texts).transform(texts)

    @property
    def feature_names(self) -> List[str]:
        """Daftar nama fitur hasil fit."""
        self._check_fitted()
        return list(self._vectorizer.get_feature_names_out())

    @property
    def n_features(self) -> int:
        """Jumlah fitur."""
        return len(self.feature_names)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "TFIDFVectorizer belum di-fit. Panggil .fit() atau .fit_transform() dulu."
            )

    def __repr__(self) -> str:
        return (
            f"TFIDFVectorizer("
            f"max_features={self.max_features}, "
            f"ngram_range={self.ngram_range}, "
            f"min_df={self.min_df}, "
            f"max_df={self.max_df})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3. PreprocessingPipeline
#    Menggabungkan TextPreprocessor + TFIDFVectorizer menjadi satu pipeline.
# ══════════════════════════════════════════════════════════════════════════════
class PreprocessingPipeline:
    """
    Pipeline lengkap: CSV/DataFrame → teks bersih → fitur TF-IDF → DataFrame siap PyCaret.
    """

    def __init__(
        self,
        text_col: str = "review_text",
        target_col: str = "sentiment",
        preprocessor: Optional[TextPreprocessor] = None,
        vectorizer: Optional[TFIDFVectorizer] = None,
        slang_dict: Optional[Union[dict, str, Path]] = None,
    ) -> None:
        self.text_col = text_col
        self.target_col = target_col
        self.slang_dict = slang_dict
        self.preprocessor = preprocessor or TextPreprocessor(slang_dict=slang_dict)
        self.vectorizer = vectorizer or TFIDFVectorizer()
        self._is_fitted = False

    def fit_transform(
        self,
        source: Union[str, Path, pd.DataFrame],
        save_cleaned: bool = False,
        cleaned_path: Union[str, Path] = "data/processed/cleaned_text.csv"
    ) -> pd.DataFrame:
        """Fit pipeline dan transform data sekaligus."""
        df = self._load(source)
        texts, labels = self._split_text_label(df)

        cleaned = self.preprocessor.transform(texts)

        if save_cleaned:
            pd.DataFrame({"clean_text": cleaned, self.target_col: labels}).to_csv(
                cleaned_path, index=False
            )
            logger.info("Teks bersih disimpan ke %s", cleaned_path)

        X_df = self.vectorizer.fit_transform(cleaned)
        X_df[self.target_col] = labels.reset_index(drop=True)

        self._is_fitted = True
        logger.info(
            "Pipeline selesai. Output shape: %s | Fitur: %d",
            X_df.shape,
            self.vectorizer.n_features,
        )
        return X_df

    def transform(
        self,
        source: Union[str, Path, pd.DataFrame],
        include_label: bool = True,
    ) -> pd.DataFrame:
        """Transform data baru menggunakan pipeline yang sudah di-fit."""
        self._check_fitted()
        df = self._load(source)
        texts = df[self.text_col]
        cleaned = self.preprocessor.transform(texts)
        X_df = self.vectorizer.transform(cleaned)

        if include_label and self.target_col in df.columns:
            X_df[self.target_col] = df[self.target_col].reset_index(drop=True)

        return X_df

    def transform_raw(self, texts: Union[str, List[str]]) -> pd.DataFrame:
        """Transform teks mentah (tanpa label)."""
        self._check_fitted()
        if isinstance(texts, str):
            texts = [texts]
        cleaned = self.preprocessor.transform(texts)
        return self.vectorizer.transform(cleaned)

    def save(self, path: Union[str, Path] = "pipeline.pkl") -> None:
        """Simpan seluruh pipeline ke file pickle."""
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Pipeline disimpan ke: %s", path.resolve())

    @classmethod
    def load(cls, path: Union[str, Path] = "pipeline.pkl") -> "PreprocessingPipeline":
        """Muat pipeline dari file pickle."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")
        with open(path, "rb") as f:
            obj = _PreprocessingUnpickler(f).load()
        logger.info("Pipeline dimuat dari: %s", path.resolve())
        return obj

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def feature_names(self) -> List[str]:
        self._check_fitted()
        return self.vectorizer.feature_names

    @property
    def n_features(self) -> int:
        self._check_fitted()
        return self.vectorizer.n_features

    def summary(self) -> None:
        """Cetak ringkasan konfigurasi pipeline."""
        print("=" * 55)
        print("         PREPROCESSING PIPELINE SUMMARY")
        print("=" * 55)
        print(f"  Kolom teks      : {self.text_col}")
        print(f"  Kolom target    : {self.target_col}")
        print(f"  Status          : {'✅ Fitted' if self._is_fitted else '⏳ Belum di-fit'}")
        print()
        print(f"  [TextPreprocessor]")
        print(f"    Lowercase              : {self.preprocessor.lowercase}")
        print(f"    Min panjang token      : {self.preprocessor.min_token_length}")
        print()
        print(f"  [TFIDFVectorizer]")
        print(f"    Max features           : {self.vectorizer.max_features}")
        print(f"    N-gram range           : {self.vectorizer.ngram_range}")
        if self._is_fitted:
            print(f"    Fitur aktual           : {self.vectorizer.n_features}")
        print("=" * 55)

    def _load(self, source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            return source.copy()
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")
        logger.info("Memuat data dari: %s", path.resolve())
        return pd.read_csv(path)

    def _split_text_label(self, df: pd.DataFrame):
        if self.text_col not in df.columns:
            raise ValueError(f"Kolom teks '{self.text_col}' tidak ditemukan.")
        if self.target_col not in df.columns:
            raise ValueError(f"Kolom target '{self.target_col}' tidak ditemukan.")
        return df[self.text_col], df[self.target_col]

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Pipeline belum di-fit.")

    def __repr__(self) -> str:
        return f"PreprocessingPipeline(text_col='{self.text_col}', target_col='{self.target_col}', fitted={self._is_fitted})"


if __name__ == "__main__":
    pipeline = PreprocessingPipeline(slang_dict="slang-indo.csv")
    pipeline.summary()
    csv_path = Path("dataset.csv")
    if csv_path.exists():
        X_df = pipeline.fit_transform(csv_path)
        print(f"Output shape: {X_df.shape}")
        pipeline.save("pipeline.pkl")
