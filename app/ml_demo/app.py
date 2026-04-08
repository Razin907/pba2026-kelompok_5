import streamlit as st
import pandas as pd
from src.preprocessing import PreprocessingPipeline
from src.utils import SentimentClassifier

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="E-Commerce Sentiment Analysis",
    page_icon="🛍️",
    layout="centered"
)

# --- Header ---
st.title("🛍️ E-Commerce Sentiment Analysis")
st.markdown("Aplikasi untuk memprediksi sentimen ulasan produk menggunakan Machine Learning.")

# --- Load Model & Pipeline ---
# Menggunakan st.cache_resource agar model tidak di-load ulang setiap kali ada input
@st.cache_resource
def load_system():
    try:
        prep = PreprocessingPipeline.load("models/pipeline.pkl")
        clf = SentimentClassifier.load("models/model.pkl")
        return prep, clf
    except FileNotFoundError as e:
        return None, None

preprocessor, classifier = load_system()

if preprocessor is None or classifier is None:
    st.error("Model atau Pipeline belum tersedia. Mohon jalankan skrip `main.py` terlebih dahulu untuk melatih dan menghasilkan file model.")
    st.stop()

# --- Area Input ---
st.write("### Masukkan Ulasan Anda:")
user_input = st.text_area("Ketik teks ulasan produk di sini (Contoh: 'Barang ini sangat bagus dan sesuai deskripsi!')", height=150)

if st.button("Analisis Sentimen", type="primary"):
    if user_input.strip() == "":
        st.warning("Mohon masukkan teks ulasan terlebih dahulu.")
    else:
        with st.spinner("Menganalisis..."):
            # Lakukan Prediksi
            hasil_df = classifier.predict_text([user_input], preprocessor)
            
            # Ambil output
            pred_label = hasil_df['prediction_label'].iloc[0]
            pred_score = hasil_df['prediction_score'].iloc[0]
            sentiment_text = hasil_df['sentiment'].iloc[0]

            # --- Tampilan Hasil ---
            st.write("---")
            st.write("### Hasil Klasifikasi")
            
            if pred_label == 1:
                st.success(f"**{sentiment_text}**")
            else:
                st.error(f"**{sentiment_text}**")
                
            st.info(f"**Confidence Score (Probabilitas Kelas Positif):** {pred_score:.2%}")
            
            # Menampilkan detail Dataframe Opsional
            with st.expander("Lihat Detail Ekstraksi Vektor"):
                df_tfidf = preprocessor.transform_raw([user_input])
                st.dataframe(df_tfidf.replace(0.0, pd.NA).dropna(axis=1)) # Tampilkan term yang aktif saja
