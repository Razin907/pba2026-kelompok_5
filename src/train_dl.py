"""
train_dl.py
===========
Pipeline Deep Learning untuk klasifikasi sentimen ulasan e-commerce
Bahasa Indonesia menggunakan arsitektur BiLSTM + Attention (PyTorch).

Arsitektur: Embedding → BiLSTM (2 layer) → Attention → FC → Softmax
Parameter : ~4.84 juta
Framework : PyTorch

Penggunaan cepat:
    python -m src.train_dl
    # atau dari root proyek:
    py src/train_dl.py
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 0. Konfigurasi
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class DLConfig:
    """
    Konfigurasi lengkap untuk eksperimen BiLSTM + Attention.

    Parameters
    ----------
    data_path      : Path ke cleaned_text.csv
    model_save_dir : Direktori penyimpanan model dan artefak
    vocab_size     : Ukuran vocabulary maksimum
    max_len        : Panjang maksimum token per kalimat (padding/truncation)
    embed_dim      : Dimensi embedding
    hidden_dim1    : Hidden unit BiLSTM layer 1 (tiap arah)
    hidden_dim2    : Hidden unit BiLSTM layer 2 (tiap arah)
    num_classes    : Jumlah kelas (2 = biner: positif/negatif)
    dropout        : Dropout rate
    batch_size     : Ukuran batch training
    num_epochs     : Jumlah epoch maksimum
    lr             : Learning rate awal
    patience       : Early stopping patience (epoch)
    seed           : Random seed untuk reproducibility
    test_size      : Proporsi data validasi
    """

    data_path: str      = "data/processed/cleaned_text.csv"
    model_save_dir: str = "models"

    # Arsitektur model
    vocab_size:  int   = 30_000
    max_len:     int   = 128
    embed_dim:   int   = 128
    hidden_dim1: int   = 256
    hidden_dim2: int   = 128
    num_classes: int   = 2
    dropout:     float = 0.3

    # Training
    batch_size:  int   = 64
    num_epochs:  int   = 15
    lr:          float = 1e-3
    patience:    int   = 3
    test_size:   float = 0.2
    seed:        int   = 42


# ══════════════════════════════════════════════════════════════════════════════
# 1. Vocabulary Builder
# ══════════════════════════════════════════════════════════════════════════════
class Vocabulary:
    """
    Membangun dan menyimpan mapping kata → indeks dari kumpulan teks.

    Token spesial:
        <PAD> = 0  (padding)
        <UNK> = 1  (unknown / out-of-vocabulary)
    """

    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, max_size: int = 30_000) -> None:
        self.max_size = max_size
        self.word2idx: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word: Dict[int, str] = {0: "<PAD>", 1: "<UNK>"}

    def build(self, texts: List[str]) -> "Vocabulary":
        """Fit vocabulary dari daftar teks."""
        counter: Counter = Counter()
        for text in texts:
            counter.update(str(text).split())

        logger.info("Total token unik di korpus: %d", len(counter))

        for word, _ in counter.most_common(self.max_size - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx]  = word

        logger.info("Vocabulary dibangun: %d token (max_size=%d)", len(self.word2idx), self.max_size)
        return self

    def encode(self, text: str, max_len: int = 128) -> List[int]:
        """Konversi teks menjadi list indeks integer dengan padding/truncation."""
        tokens = str(text).split()[:max_len]
        ids    = [self.word2idx.get(t, self.UNK_IDX) for t in tokens]
        ids   += [self.PAD_IDX] * (max_len - len(ids))
        return ids

    def save(self, path: str) -> None:
        """Simpan vocabulary ke file JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f, ensure_ascii=False)
        logger.info("Vocabulary disimpan ke: %s", path)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """Muat vocabulary dari file JSON."""
        with open(path, "r", encoding="utf-8") as f:
            word2idx = json.load(f)
        vocab          = cls(max_size=len(word2idx))
        vocab.word2idx = word2idx
        vocab.idx2word = {v: k for k, v in word2idx.items()}
        logger.info("Vocabulary dimuat: %d token dari %s", len(word2idx), path)
        return vocab

    def __len__(self) -> int:
        return len(self.word2idx)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Dataset PyTorch
# ══════════════════════════════════════════════════════════════════════════════
class SentimentDataset(Dataset):
    """
    PyTorch Dataset untuk data teks sentimen.

    Parameters
    ----------
    texts  : List teks bersih
    labels : List label integer (0 atau 1)
    vocab  : Objek Vocabulary yang sudah di-fit
    max_len: Panjang sekuens (padding/truncation)
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        vocab: Vocabulary,
        max_len: int = 128,
    ) -> None:
        self.X = torch.tensor(
            [vocab.encode(t, max_len) for t in texts], dtype=torch.long
        )
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ══════════════════════════════════════════════════════════════════════════════
# 3. Model BiLSTM + Attention
# ══════════════════════════════════════════════════════════════════════════════
class BiLSTMAttention(nn.Module):
    """
    Model klasifikasi teks dengan arsitektur BiLSTM + Attention.

    Alur:
        Input (token IDs)
            ↓
        Embedding (vocab_size × embed_dim)
            ↓
        Dropout
            ↓
        BiLSTM Layer 1 (embed_dim → hidden_dim1 × 2)
            ↓
        BiLSTM Layer 2 (hidden_dim1×2 → hidden_dim2 × 2)
            ↓
        Attention Pooling  ← bobot softmax tiap token
            ↓
        Dropout
            ↓
        FC (hidden_dim2×2 → 64 → num_classes)

    Estimasi parameter (~4.84 juta):
        Embedding  : 30.000 × 128          = 3.840.000
        BiLSTM-1   : 4×(128×256+256×256+256)×2  ≈ 526.336
        BiLSTM-2   : 4×(512×128+128×128+128)×2  ≈ 394.240
        Attention  : 256 × 1 + 1           =     257
        FC (256→64): 256×64 + 64           =  16.448
        FC (64→2)  : 64×2 + 2             =     130
        Total      :                       ≈ 4.777.411 ✅

    Parameters
    ----------
    vocab_size  : Ukuran vocabulary
    embed_dim   : Dimensi embedding
    hidden_dim1 : Hidden unit BiLSTM layer 1
    hidden_dim2 : Hidden unit BiLSTM layer 2
    num_classes : Jumlah kelas output
    dropout     : Dropout rate
    """

    def __init__(
        self,
        vocab_size:  int   = 30_000,
        embed_dim:   int   = 128,
        hidden_dim1: int   = 256,
        hidden_dim2: int   = 128,
        num_classes: int   = 2,
        dropout:     float = 0.3,
    ) -> None:
        super().__init__()


        # Embedding layer (padding_idx=0 agar <PAD> tidak ikut gradient)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)


        # BiLSTM layer 1
        self.lstm1 = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim1,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,   # dropout dikelola manual di luar LSTM
        )


        # BiLSTM layer 2
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim1 * 2,
            hidden_size=hidden_dim2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )


        # Attention layer — belajar bobot kepentingan tiap posisi
        self.attention = nn.Linear(hidden_dim2 * 2, 1, bias=True)


        self.dropout = nn.Dropout(dropout)


        # Fully Connected classifier
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim2 * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.


        Parameters
        ----------
        x : torch.Tensor — shape (batch, seq_len) berisi token IDs


        Returns
        -------
        torch.Tensor — shape (batch, num_classes) logits
        """
        # Embedding + dropout
        emb = self.dropout(self.embedding(x))        # (B, L, E)


        # BiLSTM layer 1
        out1, _ = self.lstm1(emb)                    # (B, L, 2*H1)
        out1    = self.dropout(out1)


        # BiLSTM layer 2
        out2, _ = self.lstm2(out1)                   # (B, L, 2*H2)
        out2    = self.dropout(out2)


        # Attention: hitung skor tiap posisi, softmax, weighted sum
        attn_scores  = self.attention(out2)          # (B, L, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, L, 1)
        context      = (attn_weights * out2).sum(dim=1)   # (B, 2*H2)


        # Klasifikasi
        logits = self.fc(self.dropout(context))      # (B, num_classes)
        return logits




# ══════════════════════════════════════════════════════════════════════════════
# 4. Utilitas Training
# ══════════════════════════════════════════════════════════════════════════════
def count_parameters(model: nn.Module) -> int:
    """Hitung jumlah parameter trainable pada model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




def set_seed(seed: int) -> None:
    """Set random seed untuk reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Training satu epoch.


    Returns
    -------
    (avg_loss, accuracy) pada data training epoch ini.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0


    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)


        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()


        # Gradient clipping — mencegah exploding gradient
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


        optimizer.step()


        total_loss += loss.item() * len(y_batch)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y_batch).sum().item()
        total      += len(y_batch)


    return total_loss / total, correct / total




@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluasi model pada loader.


    Returns
    -------
    (avg_loss, accuracy, all_preds, all_labels)
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []


    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)


        logits = model(X_batch)
        loss   = criterion(logits, y_batch)


        total_loss += loss.item() * len(y_batch)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y_batch).sum().item()
        total      += len(y_batch)


        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())


    return (
        total_loss / total,
        correct / total,
        np.array(all_preds),
        np.array(all_labels),
    )




# ══════════════════════════════════════════════════════════════════════════════
# 5. Pipeline Training Utama
# ══════════════════════════════════════════════════════════════════════════════
def run_training(cfg: DLConfig) -> None:
    """
    Jalankan pipeline training BiLSTM + Attention end-to-end.


    Alur:
        Load data → Build vocab → Split data → Build DataLoader
        → Build model → Training loop → Evaluasi final → Simpan artefak
    """
    set_seed(cfg.seed)


    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 55)
    logger.info("  PIPELINE DEEP LEARNING — BiLSTM + Attention")
    logger.info("=" * 55)
    logger.info("Device  : %s", DEVICE)
    logger.info("Config  : %s", cfg)


    save_dir = Path(cfg.model_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)


    # ── [1] Load data ────────────────────────────────────────────────────────
    logger.info("\n[Tahap 1] Memuat data...")
    df = pd.read_csv(cfg.data_path)
    df["clean_text"] = df["clean_text"].fillna("").astype(str)


    texts  = df["clean_text"].tolist()
    labels = df["sentiment"].tolist()


    logger.info("Jumlah sampel : %d", len(df))
    logger.info("Distribusi    : %s", dict(Counter(labels)))


    # ── [2] Build vocabulary ─────────────────────────────────────────────────
    logger.info("\n[Tahap 2] Membangun vocabulary...")
    vocab = Vocabulary(max_size=cfg.vocab_size).build(texts)
    vocab.save(str(save_dir / "vocab_dl.json"))


    # ── [3] Split data ───────────────────────────────────────────────────────
    logger.info("\n[Tahap 3] Membagi data (train/val = %.0f%%/%.0f%%)...",
                (1 - cfg.test_size) * 100, cfg.test_size * 100)


    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=labels,
    )
    logger.info("Train: %d | Val: %d", len(X_train), len(X_val))


    # ── [4] DataLoader ───────────────────────────────────────────────────────
    logger.info("\n[Tahap 4] Membuat DataLoader...")
    train_ds = SentimentDataset(X_train, y_train, vocab, max_len=cfg.max_len)
    val_ds   = SentimentDataset(X_val,   y_val,   vocab, max_len=cfg.max_len)


    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size * 2, shuffle=False, num_workers=0)


    # ── [5] Build model ──────────────────────────────────────────────────────
    logger.info("\n[Tahap 5] Membangun model BiLSTM + Attention...")
    model = BiLSTMAttention(
        vocab_size  = len(vocab),
        embed_dim   = cfg.embed_dim,
        hidden_dim1 = cfg.hidden_dim1,
        hidden_dim2 = cfg.hidden_dim2,
        num_classes = cfg.num_classes,
        dropout     = cfg.dropout,
    ).to(DEVICE)


    n_params = count_parameters(model)
    logger.info("Total parameter trainable : %s", f"{n_params:,}")
    assert n_params <= 10_000_000, (
        f"Model melebihi batas 10 juta parameter! ({n_params:,})"
    )

     # ── Optimizer & Scheduler ────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )


    # ── [6] Training loop ────────────────────────────────────────────────────
    logger.info("\n[Tahap 6] Memulai training (%d epoch)...", cfg.num_epochs)
    logger.info("-" * 75)
    logger.info("%-6s %-12s %-12s %-12s %-12s %-10s",
                "Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "LR")
    logger.info("-" * 75)


    best_val_acc    = 0.0
    best_val_f1     = 0.0
    patience_counter = 0
    history: List[dict] = []


    for epoch in range(1, cfg.num_epochs + 1):
        t_start = time.time()


        # Training
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )


        # Validasi
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, criterion, DEVICE
        )
        val_f1 = f1_score(val_labels, val_preds, average="weighted")


        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t_start


        logger.info(
            "%-6d %-12.4f %-12.4f %-12.4f %-12.4f %-10.6f (%.1fs)",
            epoch, train_loss, train_acc, val_loss, val_acc, current_lr, elapsed
        )


        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc, 4),
            "val_loss":   round(val_loss, 4),
            "val_acc":    round(val_acc, 4),
            "val_f1":     round(val_f1, 4),
        })


        # Simpan model terbaik berdasarkan val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1  = val_f1
            torch.save(model.state_dict(), str(save_dir / "bilstm_attention.pt"))
            logger.info("  ✅ Model terbaik tersimpan (val_acc=%.4f, val_f1=%.4f)",
                        best_val_acc, best_val_f1)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                logger.info(
                    "Early stopping: tidak ada peningkatan selama %d epoch.",
                    cfg.patience
                )
                break


    # ── [7] Evaluasi final ───────────────────────────────────────────────────
    logger.info("\n[Tahap 7] Evaluasi final pada validation set...")


    # Load bobot terbaik
    model.load_state_dict(
        torch.load(str(save_dir / "bilstm_attention.pt"), map_location=DEVICE)
    )


    _, final_acc, final_preds, final_labels = evaluate(
        model, val_loader, criterion, DEVICE
    )


    report = classification_report(
        final_labels, final_preds,
        target_names=["Negatif (0)", "Positif (1)"],
        digits=4,
    )
    cm = confusion_matrix(final_labels, final_preds)


    logger.info("\n%s", "=" * 55)
    logger.info("  HASIL EVALUASI FINAL")
    logger.info("=" * 55)
    logger.info("Accuracy     : %.4f", final_acc)
    logger.info("F1 (weighted): %.4f", best_val_f1)
    logger.info("\nClassification Report:\n%s", report)
    logger.info("Confusion Matrix:\n%s", cm)


    # ── [8] Simpan artefak ───────────────────────────────────────────────────
    logger.info("\n[Tahap 8] Menyimpan artefak...")


    # Training history
    history_path = str(save_dir / "bilstm_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


    # Config
    config_path = str(save_dir / "bilstm_config.json")
    with open(config_path, "w") as f:
        json.dump(cfg.__dict__, f, indent=2)


    logger.info("Model       : %s", str(save_dir / "bilstm_attention.pt"))
    logger.info("Vocabulary  : %s", str(save_dir / "vocab_dl.json"))
    logger.info("History     : %s", history_path)
    logger.info("Config      : %s", config_path)
    logger.info("\n" + "=" * 55)
    logger.info("  PIPELINE SELESAI ✅")
    logger.info("  Best Val Accuracy : %.4f", best_val_acc)
    logger.info("  Best Val F1       : %.4f", best_val_f1)
    logger.info("  Total Parameter   : %s", f"{n_params:,}")
    logger.info("=" * 55)




# ══════════════════════════════════════════════════════════════════════════════
# 6. Inference Helper
# ══════════════════════════════════════════════════════════════════════════════
class DLPredictor:
    """
    Helper class untuk inference menggunakan model BiLSTM + Attention
    yang sudah dilatih.


    Penggunaan:
        predictor = DLPredictor.load("models")
        hasil = predictor.predict(["barang zonk", "pengiriman cepat"])
        print(hasil)
    """


    LABEL_MAP = {0: "NEGATIF 👎", 1: "POSITIF 👍"}


    def __init__(
        self,
        model: BiLSTMAttention,
        vocab: Vocabulary,
        cfg: DLConfig,
        device: torch.device,
    ) -> None:
        self.model  = model
        self.vocab  = vocab
        self.cfg    = cfg
        self.device = device


    @classmethod
    def load(cls, model_dir: str = "models") -> "DLPredictor":
        """
        Muat model, vocab, dan config dari direktori.


        Parameters
        ----------
        model_dir : Direktori tempat artefak disimpan.
        """
        mdir = Path(model_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # Load config
        with open(mdir / "bilstm_config.json") as f:
            cfg_dict = json.load(f)
        cfg = DLConfig(**cfg_dict)


        # Load vocabulary
        vocab = Vocabulary.load(str(mdir / "vocab_dl.json"))


        # Load model
        model = BiLSTMAttention(
            vocab_size  = len(vocab),
            embed_dim   = cfg.embed_dim,
            hidden_dim1 = cfg.hidden_dim1,
            hidden_dim2 = cfg.hidden_dim2,
            num_classes = cfg.num_classes,
            dropout     = cfg.dropout,
        )
        model.load_state_dict(
            torch.load(str(mdir / "bilstm_attention.pt"), map_location=device)
        )
        model.to(device).eval()


        logger.info("DLPredictor siap. Device: %s | Vocab: %d token", device, len(vocab))
        return cls(model, vocab, cfg, device)


    @torch.no_grad()
    def predict(self, texts: List[str]) -> pd.DataFrame:
        """
        Prediksi sentimen dari list teks mentah.


        Parameters
        ----------
        texts : List of str — teks yang ingin diprediksi


        Returns
        -------
        pd.DataFrame dengan kolom:
            - teks           : input asli
            - label          : 0 (Negatif) atau 1 (Positif)
            - sentimen       : label teks (POSITIF / NEGATIF)
            - prob_negatif   : probabilitas kelas 0
            - prob_positif   : probabilitas kelas 1
        """
        if isinstance(texts, str):
            texts = [texts]


        # Encode & buat tensor
        ids = torch.tensor(
            [self.vocab.encode(t, self.cfg.max_len) for t in texts],
            dtype=torch.long,
        ).to(self.device)


        # Forward pass
        logits = self.model(ids)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = probs.argmax(axis=1)


        return pd.DataFrame({
            "teks":         texts,
            "label":        preds,
            "sentimen":     [self.LABEL_MAP[p] for p in preds],
            "prob_negatif": probs[:, 0].round(4),
            "prob_positif": probs[:, 1].round(4),
        })




# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    cfg = DLConfig()
    run_training(cfg)
