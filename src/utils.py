"""
klasifikasi.py
==============
Pipeline klasifikasi sentimen Bahasa Indonesia berbasis PyCaret.
Dirancang untuk bekerja bersama preprocessing.py.

Penggunaan cepat:
    from preprocessing  import PreprocessingPipeline
    from klasifikasi    import SentimentClassifier

    # 1. Preprocessing
    prep  = PreprocessingPipeline()
    X_df  = prep.fit_transform("dataset.csv")
    prep.save("pipeline.pkl")

    # 2. Klasifikasi
    clf = SentimentClassifier()
    clf.setup(X_df)                           # inisialisasi PyCaret
    clf.compare()                             # bandingkan semua model
    clf.train("lr")                           # latih model terbaik
    clf.tune()                                # tuning otomatis
    clf.evaluate()                            # laporan evaluasi
    clf.save("model.pkl")                     # simpan model

    # 3. Prediksi teks baru
    clf2  = SentimentClassifier.load("model.pkl")
    prep2 = PreprocessingPipeline.load("pipeline.pkl")
    hasil = clf2.predict_text(["barang bagus!"], prep2)
"""

from __future__ import annotations

import pickle
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. ModelConfig
#    Dataclass yang menyimpan seluruh konfigurasi eksperimen.
#    Dipisah dari class utama agar mudah diubah/disimpan/dibagikan.
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ModelConfig:
    """
    Konfigurasi lengkap untuk eksperimen klasifikasi.

    Parameters
    ----------
    target_col : str
        Nama kolom label/target di DataFrame.
    train_size : float
        Proporsi data training (0.0 – 1.0).
    fold : int
        Jumlah fold untuk stratified k-fold cross-validation.
    session_id : int
        Random seed untuk reproducibility.
    normalize : bool
        Aktifkan normalisasi fitur (MinMax).
    sort_metric : str
        Metrik utama untuk sorting hasil compare_models.
        Pilihan: 'Accuracy', 'AUC', 'F1', 'Recall', 'Precision', 'MCC'.
    n_select : int
        Jumlah model terbaik yang diambil dari compare_models.
    tune_n_iter : int
        Jumlah iterasi random search pada tune_model.
    tune_optimize : str
        Metrik yang dioptimalkan saat tuning.
    probability_threshold : float
        Ambang batas probabilitas untuk kelas positif (binary only).
    verbose : bool
        Tampilkan output PyCaret ke layar.
    experiment_name : str
        Nama eksperimen untuk MLflow tracking (opsional).
    custom_tags : dict
        Tag tambahan untuk MLflow (opsional).

    Examples
    --------
    >>> cfg = ModelConfig(fold=5, sort_metric='F1', tune_n_iter=30)
    """

    target_col: str = "sentiment"
    train_size: float = 0.80
    fold: int = 10
    session_id: int = 42
    normalize: bool = True
    sort_metric: str = "F1"
    n_select: int = 3
    tune_n_iter: int = 50
    tune_optimize: str = "F1"
    probability_threshold: float = 0.5
    verbose: bool = True
    experiment_name: str = "sentiment_classification"
    custom_tags: Dict[str, Any] = field(default_factory=dict)

    # ── Model kandidat default ─────────────────────────────────────────────
    # Dipilih berdasarkan karakteristik dataset: TF-IDF, teks pendek, balanced
    candidate_models: List[str] = field(
        default_factory=lambda: ["lr", "lightgbm", "svm", "nb", "rf"]
    )

    def __post_init__(self) -> None:
        valid_metrics = {"Accuracy", "AUC", "Recall", "Precision", "F1", "MCC", "Kappa"}
        if self.sort_metric not in valid_metrics:
            raise ValueError(
                f"sort_metric '{self.sort_metric}' tidak valid. "
                f"Pilih dari: {valid_metrics}"
            )
        if not (0.5 <= self.train_size < 1.0):
            raise ValueError("train_size harus antara 0.5 dan 1.0")
        if not (0.0 < self.probability_threshold < 1.0):
            raise ValueError("probability_threshold harus antara 0.0 dan 1.0")


# ══════════════════════════════════════════════════════════════════════════════
# 2. ExperimentResult
#    Dataclass ringan untuk menyimpan hasil setiap tahap eksperimen.
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ExperimentResult:
    """Menyimpan semua artefak hasil eksperimen."""

    compare_df: Optional[pd.DataFrame] = None       # hasil compare_models
    best_models: Optional[List[Any]] = None         # list model terbaik dari compare
    trained_model: Optional[Any] = None             # model setelah create_model
    tuned_model: Optional[Any] = None               # model setelah tune_model
    final_model: Optional[Any] = None               # model setelah finalize_model
    train_score_df: Optional[pd.DataFrame] = None   # skor training per fold
    holdout_score: Optional[pd.DataFrame] = None    # skor pada test set

    @property
    def active_model(self) -> Optional[Any]:
        """Model paling akhir yang tersedia (final > tuned > trained)."""
        return self.final_model or self.tuned_model or self.trained_model


# ══════════════════════════════════════════════════════════════════════════════
# 3. SentimentClassifier
#    Class utama — orkestrasi seluruh workflow PyCaret.
# ══════════════════════════════════════════════════════════════════════════════
class SentimentClassifier:
    """
    Orkestrasi penuh pipeline klasifikasi sentimen menggunakan PyCaret.

    Alur kerja:
        setup() → compare() → train() → tune() → evaluate() → finalize() → save()

    Parameters
    ----------
    config : ModelConfig, optional
        Konfigurasi eksperimen. Default: ModelConfig() dengan nilai optimal
        untuk dataset sentimen Indonesia.

    Examples
    --------
    # ── Alur lengkap ─────────────────────────────────────────
    from preprocessing import PreprocessingPipeline
    from klasifikasi   import SentimentClassifier, ModelConfig

    prep  = PreprocessingPipeline()
    X_df  = prep.fit_transform("dataset.csv")

    cfg = ModelConfig(fold=5, sort_metric='F1', tune_n_iter=30)
    clf = SentimentClassifier(config=cfg)

    clf.setup(X_df)
    clf.compare()
    clf.train("lr")
    clf.tune()
    clf.evaluate()
    clf.finalize()
    clf.save("model.pkl")

    # ── Prediksi teks baru ────────────────────────────────────
    loaded = SentimentClassifier.load("model.pkl")
    hasil  = loaded.predict_text(["barang zonk!"], prep)
    print(hasil)
    """

    # ID model yang tersedia di PyCaret Classification beserta nama lengkapnya
    AVAILABLE_MODELS: Dict[str, str] = {
        "lr":       "Logistic Regression",
        "knn":      "K-Nearest Neighbors",
        "nb":       "Naive Bayes",
        "dt":       "Decision Tree",
        "svm":      "SVM Linear",
        "rbfsvm":   "SVM RBF",
        "gpc":      "Gaussian Process",
        "mlp":      "MLP Neural Network",
        "ridge":    "Ridge Classifier",
        "rf":       "Random Forest",
        "qda":      "Quadratic Discriminant Analysis",
        "ada":      "AdaBoost",
        "gbc":      "Gradient Boosting",
        "lda":      "Linear Discriminant Analysis",
        "et":       "Extra Trees",
        "xgboost":  "XGBoost",
        "lightgbm": "LightGBM",
        "catboost": "CatBoost",
        "dummy":    "Dummy Classifier",
    }

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        self.config = config or ModelConfig()
        self.result = ExperimentResult()
        self._pycaret_setup_done = False
        self._pycaret = None   # referensi ke modul pycaret.classification

    # ══════════════════════════════════════════════════════════════════════════
    # TAHAP 1 — setup()
    # ══════════════════════════════════════════════════════════════════════════
    def setup(
        self,
        data: pd.DataFrame,
        **setup_kwargs: Any,
    ) -> "SentimentClassifier":
        """
        Inisialisasi eksperimen PyCaret.

        Memanggil pycaret.classification.setup() dengan parameter dari
        ModelConfig. Argumen tambahan dapat diteruskan via **setup_kwargs
        untuk override.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame fitur + kolom target. Output dari PreprocessingPipeline.
        **setup_kwargs
            Override parameter setup PyCaret (misal: html=True, silent=False).

        Returns
        -------
        SentimentClassifier (self) — untuk method chaining.

        Examples
        --------
        >>> clf.setup(X_df)
        >>> clf.setup(X_df, fold=5, normalize_method='zscore')
        """
        self._import_pycaret()

        logger.info("Menginisialisasi eksperimen PyCaret...")
        logger.info(
            "Data shape: %s | Target: '%s'",
            data.shape,
            self.config.target_col,
        )

        # Validasi kolom target
        if self.config.target_col not in data.columns:
            raise ValueError(
                f"Kolom target '{self.config.target_col}' tidak ditemukan. "
                f"Kolom tersedia: {list(data.columns)}"
            )

        default_params = dict(
            data=data,
            target=self.config.target_col,
            train_size=self.config.train_size,
            fold=self.config.fold,
            fold_strategy="stratifiedkfold",
            session_id=self.config.session_id,
            normalize=self.config.normalize,
            normalize_method="minmax",
            fix_imbalance=False,          # dataset sudah balanced 50/50
            verbose=self.config.verbose,
            html=False,
        )

        if self.config.custom_tags:
            default_params["experiment_custom_tags"] = self.config.custom_tags

        # setup_kwargs bisa override default_params
        final_params = {**default_params, **setup_kwargs}

        self._pycaret.setup(**final_params)
        self._pycaret_setup_done = True
        logger.info("Setup PyCaret selesai ✅")
        return self

    # ══════════════════════════════════════════════════════════════════════════
    # TAHAP 2 — compare()
    # ══════════════════════════════════════════════════════════════════════════
    def compare(
        self,
        models: Optional[List[str]] = None,
        sort: Optional[str] = None,
        n_select: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Bandingkan beberapa model sekaligus menggunakan cross-validation.

        Parameters
        ----------
        models : list of str, optional
            List ID model yang dibandingkan. Default: config.candidate_models.
        sort : str, optional
            Metrik sorting. Default: config.sort_metric.
        n_select : int, optional
            Jumlah model terbaik yang disimpan. Default: config.n_select.

        Returns
        -------
        pd.DataFrame — tabel perbandingan metrik semua model.

        Examples
        --------
        >>> df_compare = clf.compare()
        >>> df_compare = clf.compare(models=['lr', 'lightgbm'], sort='AUC')
        """
        self._check_setup()

        models   = models   or self.config.candidate_models
        sort     = sort     or self.config.sort_metric
        n_select = n_select or self.config.n_select

        logger.info(
            "Membandingkan %d model (sort by '%s', ambil %d terbaik)...",
            len(models), sort, n_select,
        )

        best_list = self._pycaret.compare_models(
            include=models,
            sort=sort,
            n_select=n_select,
            cross_validation=True,
            fold=self.config.fold,
            verbose=self.config.verbose,
        )

        # compare_models mengembalikan list atau single object tergantung n_select
        if not isinstance(best_list, list):
            best_list = [best_list]

        self.result.best_models = best_list
        self.result.compare_df  = self._pycaret.pull()

        logger.info("Compare selesai. Model terbaik: %s", type(best_list[0]).__name__)
        return self.result.compare_df

    # ══════════════════════════════════════════════════════════════════════════
    # TAHAP 3 — train()
    # ══════════════════════════════════════════════════════════════════════════
    def train(
        self,
        estimator: Union[str, Any] = "auto",
        fold: Optional[int] = None,
        return_train_score: bool = True,
        **kwargs: Any,
    ) -> Any:
        """
        Latih satu model menggunakan create_model PyCaret.

        Parameters
        ----------
        estimator : str atau objek, default 'auto'
            ID model (misal 'lr', 'lightgbm') atau objek sklearn.
            'auto' → gunakan model terbaik dari compare() jika sudah dijalankan,
                      atau 'lr' sebagai fallback.
        fold : int, optional
            Override jumlah fold. Default: config.fold.
        return_train_score : bool
            Tampilkan skor training untuk monitoring overfitting.
        **kwargs
            Hyperparameter langsung ke konstruktor model.
            Contoh: n_estimators=200, max_depth=10 untuk random forest.

        Returns
        -------
        Trained model object.

        Examples
        --------
        >>> clf.train('lr')
        >>> clf.train('lightgbm', n_estimators=300, learning_rate=0.05)
        >>> clf.train('auto')      # otomatis pilih hasil compare terbaik
        """
        self._check_setup()

        # Resolusi estimator 'auto'
        if estimator == "auto":
            if self.result.best_models:
                estimator = self.result.best_models[0]
                logger.info(
                    "Estimator 'auto' → menggunakan: %s",
                    type(estimator).__name__,
                )
            else:
                estimator = "lr"
                logger.info("Estimator 'auto' → fallback ke 'lr' (jalankan compare() dulu)")

        logger.info(
            "Melatih model: %s",
            type(estimator).__name__ if not isinstance(estimator, str) else estimator,
        )

        model = self._pycaret.create_model(
            estimator=estimator,
            fold=fold or self.config.fold,
            round=4,
            cross_validation=True,
            probability_threshold=self.config.probability_threshold,
            verbose=self.config.verbose,
            return_train_score=return_train_score,
            **kwargs,
        )

        self.result.trained_model  = model
        self.result.train_score_df = self._pycaret.pull()
        logger.info("Training selesai ✅")
        return model

    # ══════════════════════════════════════════════════════════════════════════
    # TAHAP 4 — tune()
    # ══════════════════════════════════════════════════════════════════════════
    def tune(
        self,
        model: Optional[Any] = None,
        optimize: Optional[str] = None,
        n_iter: Optional[int] = None,
        search_library: str = "scikit-learn",
        search_algorithm: str = "random",
        fold: Optional[int] = None,
        return_train_score: bool = True,
    ) -> Any:
        """
        Tuning hyperparameter otomatis menggunakan tune_model PyCaret.

        Parameters
        ----------
        model : objek model, optional
            Model yang di-tune. Default: hasil train() terakhir.
        optimize : str, optional
            Metrik yang dioptimalkan. Default: config.tune_optimize.
        n_iter : int, optional
            Jumlah iterasi random search. Default: config.tune_n_iter.
        search_library : str
            Library pencarian hyperparameter ('scikit-learn', 'optuna', 'tune-sklearn').
        search_algorithm : str
            Algoritma pencarian ('random', 'grid', 'bayesian').
        fold : int, optional
            Override jumlah fold.
        return_train_score : bool
            Sertakan skor training.

        Returns
        -------
        Tuned model object.

        Examples
        --------
        >>> clf.tune()
        >>> clf.tune(optimize='AUC', n_iter=100)
        >>> clf.tune(search_library='optuna', search_algorithm='tpe')
        """
        self._check_setup()

        model    = model    or self.result.trained_model
        optimize = optimize or self.config.tune_optimize
        n_iter   = n_iter   or self.config.tune_n_iter

        if model is None:
            raise RuntimeError("Tidak ada model untuk di-tune. Panggil train() dulu.")

        logger.info(
            "Tuning %s | optimize='%s' | n_iter=%d | algo='%s'",
            type(model).__name__, optimize, n_iter, search_algorithm,
        )

        tuned = self._pycaret.tune_model(
            estimator=model,
            optimize=optimize,
            n_iter=n_iter,
            search_library=search_library,
            search_algorithm=search_algorithm,
            fold=fold or self.config.fold,
            verbose=self.config.verbose,
            return_train_score=return_train_score,
        )

        self.result.tuned_model = tuned
        logger.info("Tuning selesai ✅  Model: %s", type(tuned).__name__)
        return tuned

    # ══════════════════════════════════════════════════════════════════════════
    # TAHAP 5 — evaluate()
    # ══════════════════════════════════════════════════════════════════════════
    def evaluate(self, model: Optional[Any] = None) -> None:
        """
        Tampilkan laporan evaluasi lengkap model.

        Menampilkan:
        - Confusion matrix
        - Classification report (precision, recall, F1 per kelas)
        - ROC-AUC curve
        - Precision-Recall curve
        - Feature importance (jika didukung model)

        Parameters
        ----------
        model : objek model, optional
            Default: model aktif terbaru (tuned > trained).

        Examples
        --------
        >>> clf.evaluate()
        """
        self._check_setup()
        model = model or self.result.active_model
        if model is None:
            raise RuntimeError("Tidak ada model. Panggil train() dulu.")

        logger.info("Menampilkan evaluasi model: %s", type(model).__name__)

        plots = ["confusion_matrix", "class_report", "auc", "pr"]

        # Feature importance hanya untuk model yang mendukung
        _supports_importance = (
            "RandomForest", "ExtraTrees", "GradientBoosting",
            "LGBMClassifier", "XGBClassifier", "CatBoost",
            "AdaBoost", "DecisionTree",
        )
        model_name = type(model).__name__
        if any(m in model_name for m in _supports_importance):
            plots.append("feature")

        for plot in plots:
            try:
                self._pycaret.plot_model(model, plot=plot, save=False)
            except Exception as exc:
                logger.warning("Plot '%s' gagal ditampilkan: %s", plot, exc)

    def score(self, model: Optional[Any] = None) -> pd.DataFrame:
        """
        Evaluasi model pada holdout test set dan kembalikan DataFrame skor.

        Parameters
        ----------
        model : objek model, optional
            Default: model aktif terbaru.

        Returns
        -------
        pd.DataFrame — metrik pada holdout set.

        Examples
        --------
        >>> df_score = clf.score()
        >>> print(df_score)
        """
        self._check_setup()
        model = model or self.result.active_model
        if model is None:
            raise RuntimeError("Tidak ada model. Panggil train() dulu.")

        logger.info("Evaluasi pada holdout test set...")
        self._pycaret.predict_model(model, verbose=self.config.verbose)
        score_df = self._pycaret.pull()
        self.result.holdout_score = score_df
        return score_df

    # ══════════════════════════════════════════════════════════════════════════
    # TAHAP 6 — finalize()
    # ══════════════════════════════════════════════════════════════════════════
    def finalize(self, model: Optional[Any] = None) -> Any:
        """
        Finalisasi model: latih ulang pada SELURUH data (train + test).

        Wajib dipanggil sebelum deployment/produksi agar model mendapatkan
        sebanyak mungkin data untuk belajar.

        Parameters
        ----------
        model : objek model, optional
            Default: model aktif terbaru (tuned > trained).

        Returns
        -------
        Finalized model object.

        Examples
        --------
        >>> clf.finalize()
        """
        self._check_setup()
        model = model or self.result.active_model
        if model is None:
            raise RuntimeError("Tidak ada model. Panggil train() dulu.")

        logger.info("Finalisasi model (latih ulang pada 100%% data)...")
        final = self._pycaret.finalize_model(model)
        self.result.final_model = final
        logger.info("Finalisasi selesai ✅")
        return final

    # ══════════════════════════════════════════════════════════════════════════
    # PREDIKSI
    # ══════════════════════════════════════════════════════════════════════════
    def predict_df(
        self,
        data: pd.DataFrame,
        model: Optional[Any] = None,
        raw_score: bool = True,
    ) -> pd.DataFrame:
        """
        Prediksi pada DataFrame yang sudah dipreproses (fitur TF-IDF).

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame fitur TF-IDF (output TFIDFVectorizer, TANPA kolom target).
        model : objek model, optional
            Default: model aktif terbaru.
        raw_score : bool
            Tampilkan probabilitas tiap kelas jika True.

        Returns
        -------
        pd.DataFrame — data original + kolom 'prediction_label' + 'prediction_score'.

        Examples
        --------
        >>> X_new = prep.transform_raw(["teks baru"])
        >>> hasil = clf.predict_df(X_new)
        """
        self._check_setup()
        model = model or self.result.active_model
        if model is None:
            raise RuntimeError("Tidak ada model. Panggil train() dulu.")

        result_df = self._pycaret.predict_model(
            model,
            data=data,
            raw_score=raw_score,
            verbose=False,
        )
        return result_df

    def predict_text(
        self,
        texts: Union[str, List[str]],
        preprocessing_pipeline: Any,
        model: Optional[Any] = None,
    ) -> pd.DataFrame:
        """
        Prediksi end-to-end dari teks mentah ke label sentimen.

        Secara internal: teks → preprocessing → TF-IDF → prediksi model.

        Parameters
        ----------
        texts : str atau list of str
            Teks mentah yang ingin diprediksi.
        preprocessing_pipeline : PreprocessingPipeline
            Instance yang sudah di-fit (muat dari .pkl atau dari sesi sebelumnya).
        model : objek model, optional
            Default: model aktif terbaru.

        Returns
        -------
        pd.DataFrame dengan kolom:
            - review_text        : teks asli
            - prediction_label   : label prediksi (0 atau 1)
            - prediction_score   : probabilitas kelas positif

        Examples
        --------
        >>> hasil = clf.predict_text(
        ...     ["barang bagus banget!", "zonk nyesel beli"],
        ...     prep
        ... )
        >>> print(hasil[['review_text', 'prediction_label', 'prediction_score']])
        """
        if isinstance(texts, str):
            texts = [texts]

        # 1. Transform teks ke fitur TF-IDF
        X_new = preprocessing_pipeline.transform_raw(texts)

        # 2. Prediksi
        pred_df = self.predict_df(X_new, model=model, raw_score=True)

        # 3. Susun output yang informatif
        output = pd.DataFrame({
            "review_text":        texts,
            "prediction_label":   pred_df["prediction_label"].values,
            "prediction_score":   pred_df.get(
                "prediction_score_1",
                pred_df.filter(like="prediction_score").iloc[:, -1]
            ).values,
            "sentiment":          pred_df["prediction_label"].map(
                {0: "NEGATIF 👎", 1: "POSITIF 👍"}
            ).values,
        })
        return output

    # ══════════════════════════════════════════════════════════════════════════
    # SIMPAN & MUAT
    # ══════════════════════════════════════════════════════════════════════════
    def save(
        self,
        path: Union[str, Path] = "model.pkl",
        save_pycaret_model: bool = True,
    ) -> None:
        """
        Simpan SentimentClassifier ke file pickle.

        Dua file disimpan:
        - <path>           → objek SentimentClassifier (config, result, dsb.)
        - <path>_pycaret   → model PyCaret native (via save_model)

        Parameters
        ----------
        path : str atau Path
        save_pycaret_model : bool
            Jika True, juga simpan model native PyCaret (lebih portabel).

        Examples
        --------
        >>> clf.save("model.pkl")
        """
        path = Path(path)

        # Simpan objek SentimentClassifier
        # (tanpa referensi modul pycaret agar lebih bersih)
        _pycaret_ref = self._pycaret
        self._pycaret = None

        with open(path, "wb") as f:
            pickle.dump(self, f)

        self._pycaret = _pycaret_ref
        logger.info("SentimentClassifier disimpan ke: %s", path.resolve())

        # Simpan model PyCaret native (format joblib, lebih stabil lintas versi)
        if save_pycaret_model and self.result.active_model is not None:
            pycaret_path = str(path).replace(".pkl", "_pycaret")
            self._pycaret.save_model(self.result.active_model, pycaret_path)
            logger.info("Model PyCaret native disimpan ke: %s", pycaret_path + ".pkl")

    @classmethod
    def load(cls, path: Union[str, Path] = "model.pkl") -> "SentimentClassifier":
        """
        Muat SentimentClassifier dari file pickle.

        Parameters
        ----------
        path : str atau Path

        Returns
        -------
        SentimentClassifier

        Examples
        --------
        >>> clf = SentimentClassifier.load("model.pkl")
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")

        with open(path, "rb") as f:
            obj = pickle.load(f)

        # Re-import modul pycaret setelah load
        obj._import_pycaret()
        logger.info("SentimentClassifier dimuat dari: %s", path.resolve())
        return obj

    # ══════════════════════════════════════════════════════════════════════════
    # UTILITAS
    # ══════════════════════════════════════════════════════════════════════════
    def summary(self) -> None:
        """Cetak ringkasan konfigurasi dan status eksperimen."""
        model_name = (
            type(self.result.active_model).__name__
            if self.result.active_model else "Belum ada"
        )
        print("=" * 58)
        print("         SENTIMENT CLASSIFIER SUMMARY")
        print("=" * 58)
        print(f"  Kolom target         : {self.config.target_col}")
        print(f"  Train size           : {self.config.train_size:.0%}")
        print(f"  Fold CV              : {self.config.fold}")
        print(f"  Session ID           : {self.config.session_id}")
        print(f"  Normalisasi          : {self.config.normalize}")
        print(f"  Prob. threshold      : {self.config.probability_threshold}")
        print()
        print(f"  [Comparing]")
        print(f"    Kandidat model     : {self.config.candidate_models}")
        print(f"    Sort metrik        : {self.config.sort_metric}")
        print(f"    N select           : {self.config.n_select}")
        print()
        print(f"  [Tuning]")
        print(f"    Optimize metrik    : {self.config.tune_optimize}")
        print(f"    N iter             : {self.config.tune_n_iter}")
        print()
        print(f"  [Status Eksperimen]")
        print(f"    Setup done         : {'✅' if self._pycaret_setup_done else '❌'}")
        print(f"    Compare done       : {'✅' if self.result.compare_df is not None else '❌'}")
        print(f"    Trained model      : {'✅' if self.result.trained_model else '❌'}")
        print(f"    Tuned model        : {'✅' if self.result.tuned_model else '❌'}")
        print(f"    Final model        : {'✅' if self.result.final_model else '❌'}")
        print(f"    Model aktif        : {model_name}")
        print("=" * 58)

    @staticmethod
    def list_models() -> pd.DataFrame:
        """
        Tampilkan semua model yang tersedia beserta nama lengkapnya.

        Returns
        -------
        pd.DataFrame

        Examples
        --------
        >>> SentimentClassifier.list_models()
        """
        df = pd.DataFrame(
            list(SentimentClassifier.AVAILABLE_MODELS.items()),
            columns=["ID", "Nama Model"],
        )
        print(df.to_string(index=False))
        return df

    # ══════════════════════════════════════════════════════════════════════════
    # PRIVATE
    # ══════════════════════════════════════════════════════════════════════════
    def _import_pycaret(self) -> None:
        """Lazy import PyCaret agar file bisa di-import tanpa PyCaret terinstall."""
        try:
            import pycaret.classification as _pc
            self._pycaret = _pc
        except ImportError as exc:
            raise ImportError(
                "PyCaret belum terinstall. Jalankan:\n"
                "  pip install pycaret"
            ) from exc

    def _check_setup(self) -> None:
        """Pastikan setup() sudah dipanggil sebelum operasi apapun."""
        if not self._pycaret_setup_done:
            raise RuntimeError(
                "Eksperimen belum diinisialisasi. Panggil setup(data) terlebih dahulu."
            )

    def __repr__(self) -> str:
        model_name = (
            type(self.result.active_model).__name__
            if self.result.active_model else "None"
        )
        return (
            f"SentimentClassifier("
            f"target='{self.config.target_col}', "
            f"fold={self.config.fold}, "
            f"setup_done={self._pycaret_setup_done}, "
            f"active_model={model_name})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Contoh penggunaan (jalankan langsung: python klasifikasi.py)
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from pathlib import Path

    # ── Info model tersedia ───────────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("Model tersedia di SentimentClassifier:")
    print("=" * 58)
    SentimentClassifier.list_models()

    # ── Summary konfigurasi default ───────────────────────────────────────────
    print()
    clf = SentimentClassifier()
    clf.summary()

    # ── Contoh konfigurasi kustom ─────────────────────────────────────────────
    print("\nContoh konfigurasi kustom:")
    print("-" * 40)
    cfg = ModelConfig(
        fold=5,
        sort_metric="F1",
        tune_n_iter=30,
        probability_threshold=0.45,
        candidate_models=["lr", "lightgbm", "svm"],
    )
    clf_custom = SentimentClassifier(config=cfg)
    clf_custom.summary()

    # ── Alur lengkap (hanya jika dataset & preprocessing.py tersedia) ─────────
    dataset_path   = Path("dataset.csv")
    pipeline_path  = Path("pipeline.pkl")

    if dataset_path.exists():
        print("\n" + "=" * 58)
        print("MENJALANKAN ALUR LENGKAP")
        print("=" * 58)

        # Import preprocessing
        try:
            from preprocessing import PreprocessingPipeline

            # 1. Preprocessing
            if pipeline_path.exists():
                prep = PreprocessingPipeline.load(pipeline_path)
                X_df = prep.fit_transform(dataset_path)
            else:
                prep = PreprocessingPipeline()
                X_df = prep.fit_transform(dataset_path)
                prep.save(pipeline_path)

            print(f"\nData siap. Shape: {X_df.shape}")

            # 2. Klasifikasi
            clf = SentimentClassifier()

            clf.setup(X_df)
            clf.compare()
            clf.train("auto")
            clf.tune()
            score = clf.score()
            print("\nHoldout Score:")
            print(score)
            clf.finalize()
            clf.save("model.pkl")

            # 3. Prediksi teks baru
            prep_loaded = PreprocessingPipeline.load("pipeline.pkl")
            clf_loaded  = SentimentClassifier.load("model.pkl")

            contoh = [
                "barang bagus banget rekomen bgt top markotop!",
                "zonk parah nyesel beli buang duit aja gak rekomen",
                "pengiriman cepat sesuai deskripsi puas banget",
                "kualitas jelek sekali mengecewakan",
            ]

            hasil = clf_loaded.predict_text(contoh, prep_loaded)
            print("\n" + "=" * 58)
            print("HASIL PREDIKSI:")
            print("=" * 58)
            for _, row in hasil.iterrows():
                print(f"  [{row['sentiment']}] \"{row['review_text'][:60]}\"")
                print(f"  Score: {row['prediction_score']:.4f}\n")

        except ImportError:
            print("⚠️  preprocessing.py tidak ditemukan di direktori yang sama.")
    else:
        print("\nℹ️  Letakkan 'dataset.csv' dan 'preprocessing.py' di direktori")
        print("   yang sama untuk menjalankan alur lengkap.")

    print("\n✅ Selesai.")