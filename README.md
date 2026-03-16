# Sentiment Analysis Menggunakan Library PyCaret

Team :
- Razin Hafid Hamdi (123450096)
- Ivana Margareth Hutabarat (123410028)
- Hanna Gresia Sinaga (123450038)

Project ini adalah implementasi sistem pemrosesan dan pemodelan machine learning dalam analisis teks (Sentiment Analysis) yang berbasis pada library Python `pandas` dan `pycaret`. Ekstraksi data berfokus pada dataset lokal (`dataset.csv`) yang diadaptasi dari Hugging Face [dataset-klasifikasi-sentimen-ulasan-produk](https://huggingface.co/datasets/siRendy/dataset-klasifikasi-sentimen-ulasan-produk).

## Prasyarat dan Instalasi Lingkungan Virtual (`.env`)

> [!WARNING]
> **Kompatibilitas Versi Python**: PyCaret 3.x memiliki kapabilitas dependensi ketat (khususnya dengan algoritma yang mengandalkan NumPy versi lama). Pastikan versi Python yang Anda instal di sistem/lingkungan virtual adalah di antara **Python 3.8 hingga Python 3.11**. (Python 3.12/3.13 ke atas dapat menyebabkan error saat instalasi/import library ini). 

Sebuah virtual environment yang terisolasi digunakan untuk menjalankan skrip.

1. Buat direktori kerja atau pergi ke direktori kerja yang sudah ada.
2. buat environment terlebih dahulu (jika belum ada):
   ```cmd
   python -m venv <nama_environment>
   ```
3. Aktifkan virtual environment yang telah dibuat sebelumnya:

   windows :
   ```cmd
   .\<nama_environment>\Scripts\activate
   ```
   linux/mac :
   ```cmd
   source <nama_environment>/bin/activate
   ```
5. Jika modul belum terinstal, Anda dapat menginstalnya sekaligus melalui file requirements:
   ```cmd
   pip install -r requirements.txt
   ```

## Struktur File Utama

- **`preprocessing.py`**: Mengandung kelas `DataPreprocessor` guna meload `dataset.csv` menggunakan pandas. Termasuk metode pembersihan/pre-processing teks, regex pembuangan link/url, tanda baca, standarisasi lowercase. Selain itu, ditambahkan **penanganan kata slang** menggunakan *dictionary mapping* dan **penghapusan stopwords** (bahasa Indonesia `nltk`, namun dengan pengecualian kata-kata negasi `tidak, jangan, bukan` agar makna sentimen tetap terjaga). Lemmatization dan Stemming ditiadakan melalui justifikasi performa (algoritma Sastrawi memerlukan komputasi yang sangat lama. Sekitar ~2 jam komputasi untuk dataset 19k baris ini), dan karena slang normalization seringkali memberikan performa yang cukup baik.

- **`klasifikasi.py`**: Mengandung kelas `SentimentClassifier` di mana PyCaret 3.x Environment diinisialisasi. Semua konfigurasi ML diatur disini. Termasuk train, test, juga compare models.

## Cara Penggunaan Class / API

### Preprocessing Datasets
Bisa digunakan sebagai modul (import):
```python
from Preprocessing import DataPreprocessor

# Deklarasi Object (Secara otomatis akan meload file slang-indo.csv jika ada)
preprocessor = DataPreprocessor(filepath='dataset.csv', slang_filepath='slang-indo.csv')

# Load, Bersihkan Teks, dan Kembalikan Bentuk DataFrame
df_clean = preprocessor.get_processed_data()
print(df_clean.head())
```

### Melatih Klasifikasi dan Evaluasi
Selanjutnya operasikan object dari Class *Classifier*:
```python
from klasifikasi import SentimentClassifier

# Asumsi dataset kita adalah target `sentiment`
classifier = SentimentClassifier(data=df_clean, target_col='sentiment')

# Inisialisasi Environment Machine Learning (Validation 20%)
classifier.setup_environment(test_size=0.2)

# Opsi 1: Train sebuah model (Misalnya Naive Bayes)
model_nb = classifier.train_model('nb')

# Opsi 2: Mencari otomatis seluruh model algoritma membandingkan Accuration Value
# model_terbaik = classifier.compare_all_models(sort='Accuracy', n_select=1)

# Lakukan Evaluasi Matrix (Prediksikan ke data testing/hold-out)
hasil_evaluasi = classifier.evaluate_model()

# Jika punya dataframe tak berlabel baru
# data_tak_berlabel = ...
# hasil_akhir = classifier.predict(data_tak_berlabel)
```

## Kebijakan Pembuatan File `.log`
Pada `klasifikasi.py`, sudah diatur untuk menambahkan parameter operasional argumen `system_log=False` yang ada dalam modul `setup()` milik library *pycaret*. Sehingga ketika *code running*, hal ini menjamin PyCaret tidak akan melakukan auto-output berkas logs ke dalam format ekstensi *.log* lokal yang bisa menimbulkan sampah direktori.

## Pemanfaatan Artificial Intelligence (AI) Prompting

Dalam pengembangan sistem Sentiment Analysis ini, kami menggunakan *Gemini 3.1 Pro (High) AI agents* sebagai asisten. Berikut adalah riwayat prompt utama yang digunakan selama proses pengembangan:

### Prompt 1: Setup Lingkungan dan Struktur Dasar
text
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
