<div align="center">

# 📊 E-Commerce Sentiment Analysis Pipeline
**Sistem Klasifikasi Teks Berbasis Machine Learning dengan PyCaret**

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyCaret](https://img.shields.io/badge/PyCaret-3.3.2-orange.svg)](https://pycaret.org/)
[![Pandas](https://img.shields.io/badge/pandas-data_manipulation-yellow.svg)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 🎯 Tentang Proyek

Proyek ini adalah implementasi sistem **Natural Language Processing (NLP)** yang dirancang khusus untuk menganalisis sentimen pengguna (Ulasan e-Commerce). Pipeline ini mengotomatiskan serangkaian proses mulai dari *Data Preprocessing* tingkat lanjut hingga *Model Training & Evaluation* menggunakan ekosistem terpadu **PyCaret 3.x**.

Sistem ini mengekstraksi dataset ulasan produk dari Hugging Face ([siRendy/dataset-klasifikasi-sentimen-ulasan-produk](https://huggingface.co/datasets/siRendy/dataset-klasifikasi-sentimen-ulasan-produk)) dan memprosesnya secara cepat tanpa mengonsumsi algoritma komputasi berat.

### 👥 Pengembang (Tim)
- **Razin Hafid Hamdi** (123450096)
- **Ivana Margareth Hutabarat** (123410028)
- **Hanna Gresia Sinaga** (123450038)


---

## 🏗️ Arsitektur Pipeline

Proyek ini disusun dalam pola *Object-Oriented Programming (OOP)* agar mudah digunakan ulang dan dikembangkan (Scalable).

| Modul Utama | Tanggung Jawab (Fungsi Kelas) |
| --- | --- |
| 🧹 `preprocessing.py` | Berisi kelas `PreprocessingPipeline`. Mengelola pembersihan teks (*TextPreprocessor*), normalisasi slang Indonesia via `slang-indo.csv`, dan ekstraksi fitur berbasis `TFIDFVectorizer` (N-gram support). |
| 🧠 `klasifikasi.py` | Berisi kelas `SentimentClassifier`. Bertindak sebagai _Wrapper_ library PyCaret. Bertanggung jawab mengatur `setup()` environment, perbandingan model (`compare`), training (`train`), dan evaluasi metrik (`score`). |

---

## 🚀 Panduan Instalasi (Quick Start)

Kami sangat menyarankan Anda menjalankannya pada *Virtual Environment* yang terisolasi.

> [!WARNING]
> PyCaret 3.x memiliki kompatibilitas ketat dengan iterasi dependensi di bawahnya. Anda wajib menggunakan versi **Python 3.11**. (Python 3.12+ atau 3.13+ mungkin mengalami kendala instalasi `numpy` atau `pycaret` pada sistem Windows tanpa build tools).

### 1. Buat Lingkungan Virtual (Virtual Environment)
```bash
# Membuat environment dengan Python 3.11
py -3.11 -m venv .env

# Aktivasi di Windows
.\.env\Scripts\activate
```

### 2. Instalasi Dependensi
Jalankan perintah ini untuk memasang instalasi `pandas`, `pycaret`, `scikit-learn`, dan lainnya:
```bash
pip install -r requirements.txt
```

---

## 💻 Contoh Penggunaan API (Usage)

Integrasi Pipeline Preprocessing dan Training kini dapat dilakukan secara terpadu (seperti tertera pada file `main.py`):

```python
from preprocessing import PreprocessingPipeline
from klasifikasi import SentimentClassifier

# 1. Pipeline Preprocessing & Fitur (TF-IDF)
# Mendukung normalisasi slang otomatis via file CSV
pipeline = PreprocessingPipeline(slang_dict='slang-indo.csv')
df_clean = pipeline.fit_transform('dataset.csv')

# 2. Pipeline Machine Learning (PyCaret Wrapper)
clf = SentimentClassifier()

# Inisialisasi Environment (Split Train/Test)
clf.setup(data=df_clean, train_size=0.8)

# Bandingkan model untuk mencari algoritma terbaik
best_models = clf.compare(sort='Accuracy', n_select=3)

# Latih model (otomatis menggunakan hasil terbaik dari compare)
clf.train("auto")

# 3. Evaluasi Akhir
hasil = clf.score()
print(hasil)
```

---

## 🤖 Pemanfaatan AI Prompting (Asisten Pengembangan)

Selama iterasi *Pair-Programming* pengembangan modul proyek ini, kami menggunakan **Gemini 3.1 Pro (High) AI agents** sebagai instrumen bantuan pembantu penyusunan kode. Berikut adalah histori *prompt* mentah (*Raw Prompts*) yang merepresentasikan arahan strategis arsitektur kami:

<details>
<summary><b>Lihat Histori Prompt di sini (Klik untuk memperluas)</b></summary>
<br>

**1. Setup Lingkungan dan Struktur Dasar**
```text
step by step :
1. periksa data terlebih dahulu file:dataset.csv (data ini berasal dari hugging face)
2. hapus .env yang telah tersedia dan buat yang baru dengan nama .env juga dan aktifkan
3. periksa link dokumentasi berikut terlebih dahulu -> https://pycaret.readthedocs.io/en/latest/installation.html
4. install packages pandas, pycaret, python yang compatible dengan pycaret, re, dataset dan packages yang dibutuhkan lainnya
5. periksa lagi dataset lagi file:dataset.csv
6. buat file preprocessing.py dengan class berisi atribut dan method yang dapat di panggil kedepannya
7. periksa link dokumentasi terkait klasifikasi menggunakan pycaret berikut terlebih dahulu ->https://pycaret.readthedocs.io/en/latest/api/classification.html
8. buat file klasifikasi.py dengan class berisi atribut dan method yang dapat di panggil kedepannya
9. buat dokumentasi README.md

rules :
1. jangan buat file .log apapun
2. buatkan dokumentasi singkat ("""...""") di setiap fungsi yang dibuat
```

**2. Analisis Konteks & Skema Preprocessing**
```text
lihat kembali file dataset.csv ini dan analisis preprocessing yang masih kurang dan sesuaikan dengan konteks dataset (sebagai pengetahuan dataset merupakan diambil dari hasil review sebuah produk di e commerce) , seperti :
1. penanganan slang (perlu membaca dan analisis dataset.csv terlebih dahulu)
2. lemmitazion
3. stopword
4. stemming

jika, 4 proses diatas tidak di perlukan untuk konteks dataset yang ada jangan di masukkan ke dalam preprocessing dan perbarui README.md 
```

**3. Integrasi Kamus Bahasa Slang**
```text
1. berikut dict slang-indo.csv yang berisi 2 kolom slang dan formal yang didapatkan dari hungging face
2. cocokkan dengan dataset.csv dan jadikan sebagai slang_dict
```

**4. Finalisasi Standar Reproduksi Kode**
```text
perbarui README.md dan buat requirements.txt yang berisi setiap packages yang perlu di install oleh siapa saja yang ingin mencoba pipeline ini
```
</details>

---
<div align="center">
<i>Terima Kasih! Jika Anda bagian dari tim dan ingin berkontribusi, mohon periksa file <code>CONTRIBUTING.md</code>.</i>
</div>
