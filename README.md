---
title: E Commerce Sentiment Analysis
emoji: 🛍️
colorFrom: blue
colorTo: green
sdk: gradio
python_version: 3.11
sdk_version: 4.36.1
app_file: app/ml_demo/app.py
pinned: false
---

<div align="center">

# E-Commerce Sentiment Analysis
**Sistem Klasifikasi Teks Berbasis Machine Learning dengan PyCaret**

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyCaret](https://img.shields.io/badge/PyCaret-3.3.2-orange.svg)](https://pycaret.org/)
[![Pandas](https://img.shields.io/badge/pandas-data_manipulation-yellow.svg)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

---

## Project Description

Proyek ini adalah implementasi sistem **Natural Language Processing (NLP)** yang dirancang khusus untuk menganalisis sentimen pengguna (Ulasan e-Commerce). Pipeline ini mengotomatiskan serangkaian proses mulai dari *Data Preprocessing* tingkat lanjut hingga *Model Training & Evaluation* menggunakan ekosistem terpadu **PyCaret 3.x**.

## Dataset

Dataset Link:
Sistem ini mengekstraksi dataset ulasan produk dari Hugging Face ([siRendy/dataset-klasifikasi-sentimen-ulasan-produk](https://huggingface.co/datasets/siRendy/dataset-klasifikasi-sentimen-ulasan-produk)) dan memprosesnya secara cepat tanpa mengonsumsi algoritma komputasi berat.

### Team Members

| Name                     | NIM       | GitHub Username |
| ------------------------ | --------- | --------------- |
| Razin Hafid Hamdi        | 123450096 | Razin907        |
| Ivana Margareth Hutabarat| 123410028 | -               |
| Hanna Gresia Sinaga      | 123450038 | -               |


---

## Pipeline Architecture

Proyek ini disusun dalam pola *Object-Oriented Programming (OOP)* agar modul-modul intinya mudah digunakan ulang.

| Modul Utama | Tanggung Jawab (Fungsi Kelas) |
| --- | --- |
| 🧹 `preprocessing.py` | Berisi kelas `PreprocessingPipeline`. Mengelola pembersihan teks (*TextPreprocessor*), normalisasi slang Indonesia via `slang-indo.csv`, dan ekstraksi fitur berbasis `TFIDFVectorizer` (N-gram support). |
| 🧠 `utils.py` | Berisi kelas `SentimentClassifier` dengan kapabilitas _Wrapper_ library PyCaret. Bertanggung jawab mengatur `setup()`, fitur komparasi algoritma (`compare`), serta eval dan auto-tuning. |
| 🚀 `train_ml.py` | Skrip master ekseskusi (sebelumnya `main.py`). Skrip ini yang akan dieksekusi terminal untuk menggerakan seluruh pipeline ML secara sekuensial sampai mengekspor format `.pkl` model. |

---

## Repository Structure

```text
pba2026-kelompok_5 
│
├── data
│   ├── raw
│   └── processed
│
├── notebooks
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_pycaret_model.ipynb
│   └── 04_deep_learning.ipynb
│
├── src
│   ├── preprocessing.py
│   ├── train_ml.py
│   ├── train_dl.py
│   └── utils.py
│
├── models
│
├── app
│   ├── ml_demo
│   └── dl_demo
│
├── paper
│
└── README.md
```

---

## Quick Installation

Kami sangat menyarankan Anda menjalankannya pada *Virtual Environment* yang terisolasi.

> [!WARNING]
> PyCaret 3.x memiliki kompatibilitas ketat dengan iterasi dependensi di bawahnya. Anda wajib menggunakan versi **Python antara 3.8 dan 3.11**. (Python 3.12+ belum mendukung versi PyCaret ini).

### 1. Buat Lingkungan Virtual (Virtual Environment)
```bash
# Membuat environment dengan Python 3.11
py -3.11 -m venv .env

# Aktivasi di Windows
.\.env\Scripts\activate

# Aktivasi di MacOS / Linux
source .env/bin/activate
```

### 2. Instalasi Dependensi
Jalankan perintah ini untuk memasang instalasi `pandas`, `pycaret`, `scikit-learn`, dan lainnya:
```bash
pip install -r requirements.txt
```

---

## Contoh Penggunaan API (Usage)

Integrasi Pipeline Preprocessing dan Training kini dapat dilakukan hanya dengan beberapa baris kode (seperti tertera pada file `main.py`):

```python
from Preprocessing import DataPreprocessor
from klasifikasi import SentimentClassifier

# 1. Pipeline Pembersihan Data
preprocessor = DataPreprocessor(filepath='dataset.csv', slang_filepath='slang-indo.csv')
df_clean = preprocessor.get_processed_data()

# 2. Pipeline Machine Learning
clf = SentimentClassifier(data=df_clean, target_col='sentiment')
clf.setup_environment(test_size=0.2)

# Opsi Cepat (Melatih spesifik model: regresi logistik)
model_cepat = clf.train_model('lr')

# Opsi Auto-ML (Cari model terbaik dari seluruh algoritma)
# best_model = clf.compare_all_models(sort='Accuracy', n_select=1)

# 3. Evaluasi
clf.evaluate_model()
```

---

## Deployment

Two [interactive demos will be deployed using **Hugging Face Spaces**:

* **Machine Learning Model (PyCaret)**
  - [Machine Learning](https://huggingface.co/spaces/razin-hafid/E-Commerce_Sentiment_Analysis)

* **Deep Learning Model (PyTorch)**
  - [Deep Learning](https://huggingface.co/spaces/razin-hafid/E-Commers-sentiment-analysis-DL)

---

## Pemanfaatan AI Prompting (Asisten Pengembangan)

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
</div>
