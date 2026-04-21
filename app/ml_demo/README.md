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

# Machine Learning Demo (PyCaret)

Folder ini berisi aplikasi antarmuka Gradio khusus untuk menjalankan model Traditional Machine Learning (melalui PyCaret pipeline) secara lokal di komputer Anda.

## Cara Menjalankan

1. Buka terminal dan arahkan ke folder utama proyek:
   ```bash
   cd path/to/sentiment-analysis
   ```

2. (Opsional) Gunakan Virtual Environment dengan versi Python yang kompatibel (umumnya Python 3.9 - 3.10 cocok untuk PyCaret).

3. Instal dependensi khusus untuk ML Demo:
   ```bash
   pip install -r app/ml_demo/requirements.txt
   ```

4. Jalankan aplikasi:
   ```bash
   python app/ml_demo/app.py
   ```

5. Buka tautan lokal (biasanya `http://127.0.0.1:7860`) yang muncul di terminal Anda pada browser.
