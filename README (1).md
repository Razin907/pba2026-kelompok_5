---
title: ITera NLP Hate Speech Detection
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: app/ml_demo/app.py
pinned: false
---

# NLP Text Classification Benchmark

Natural Language Processing Project — Institut Teknologi Sumatera (ITERA)

## Project Description

This project aims to compare the performance of **Machine Learning (ML)** and **Deep Learning (DL)** approaches for Natural Language Processing (NLP) tasks.

Machine Learning models will be implemented using **PyCaret AutoML**, while the Deep Learning model will be implemented using **PyTorch**.

Both approaches will be evaluated and compared on the same dataset to determine which method performs better for the selected text classification task.

---

## Team Members

| Name                     | NIM       | GitHub Username |
| ------------------------ | --------- | --------------- |
| Tanty Widiyastuti        | 123450094 | tantywidiyastuti|
| Mayada                   | 121450145 | -               |
| Adisty Syawalda Ariyanto | 121450136 | adistyS         |

---

## Dataset

Dataset used in this project will be sourced from public NLP datasets such as Kaggle, Hugging Face Datasets, or other open repositories.

Dataset Link:
(https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection?tab=readme-ov-file)

---

## Project Objectives

The objectives of this project are:

* Perform **Exploratory Data Analysis (EDA)** on the selected dataset
* Implement **Machine Learning models using PyCaret AutoML**
* Implement **Deep Learning models using PyTorch**
* Compare the performance between ML and DL models
* Deploy interactive demos using **Hugging Face Spaces**
* Publish a scientific report in **ArXiv format**

---

## Repository Structure

```
pba2026-nama-kelompok 12
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

## Machine Learning Approach

Machine Learning models will be developed using **PyCaret AutoML**.
Several algorithms will be compared automatically, and the best-performing model will be selected based on evaluation metrics.

Examples of algorithms evaluated:

* Logistic Regression
* Random Forest
* Support Vector Machine
* Gradient Boosting

---

## Deep Learning Approach

The Deep Learning model will be implemented using **PyTorch**.

Possible architectures include:

* LSTM
* GRU
* CNN for text classification
* Lightweight Transformer models

The model will be trained and evaluated using standard NLP evaluation metrics.

---

## Deployment

Two interactive demos will be deployed using **Hugging Face Spaces**:

* **Machine Learning Model (PyCaret)**
  *(link will be added later)*

* **Deep Learning Model (PyTorch)**
  *(link will be added later)*

---

## Scientific Paper

The final project report will be written in **LaTeX using ArXiv format** and will include:

* Dataset description
* Methodology
* Experiment setup
* Benchmark results
* Comparative analysis

ArXiv Link:
*(to be added)*

---

## Course Information

Course: **Pemrosesan Bahasa Alami**
Program: **Sains Data — Institut Teknologi Sumatera**
Semester: **Genap 2025/2026**

Instructor:
Martin C.T. Manullang
