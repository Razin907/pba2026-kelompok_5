import gradio as gr
import pandas as pd
from src.preprocessing import PreprocessingPipeline
from src.utils import SentimentClassifier
import os

# --- Load Model & Pipeline ---
def load_system():
    try:
        # Pipeline dan Classifier dibaca dari direktori "models/" sesuai pola deployment
        prep = PreprocessingPipeline.load("models/pipeline.pkl")
        clf = SentimentClassifier.load("models/model.pkl")
        return prep, clf
    except Exception as e:
        print(f"Error memuat model: {e}")
        return None, None

# Objek preprocessor dan classifier diload di memory sekali saja saat aplikasi start
preprocessor, classifier = load_system()

def predict_sentiment(user_input):
    # Validasi awal
    if preprocessor is None or classifier is None:
        return "⚠️ Error: Sistem klasifikasi (.pkl) gagal ter-load.", None
        
    if not user_input or str(user_input).strip() == "":
        return "⚠️ Mohon ketik ulasan terlebih dahulu pada kotak input.", None
        
    try:
        # Melakukan Prediksi
        hasil_df = classifier.predict_text([user_input], preprocessor)
        
        # Mengekstraksi kolom hasil dataframe PyCaret
        pred_label = hasil_df['prediction_label'].iloc[0]
        pred_score = float(hasil_df['prediction_score'].iloc[0]) 
        sentiment_text = hasil_df['sentiment'].iloc[0]
        
        # Format teks utama
        if pred_label == 1:
            icon = "✅ Ulasan Terindikasi Positif"
        else:
            icon = "❌ Ulasan Terindikasi Negatif"
            
        result_text = f"### {icon}\n\nNilai asli sentimen PyCaret: **{sentiment_text}**"
        
        # Format dictionary untuk Gradio Label Component (Confidence Bar)
        prob_positif = pred_score if pred_label == 1 else 1.0 - pred_score
        prob_negatif = pred_score if pred_label == 0 else 1.0 - pred_score
        
        confidence_dict = {
            "Sentimen Positif": prob_positif, 
            "Sentimen Negatif": prob_negatif
        }
                      
        return result_text, confidence_dict
        
    except Exception as e:
        return f"Terjadi kesalahan saat memproses data: {str(e)}", None

# --- Pembangunan Gradio Blocks Interface ---
with gr.Blocks(title="E-Commerce Sentiment Analysis", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛍️ E-Commerce Sentiment Analysis")
    gr.Markdown("Pakar deteksi sentimen ulasan berdaya komputasi Machine Learning (PyCaret).")
    
    with gr.Row():
        # Kolom Kiri: Input
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Masukkan Ulasan Produk (Teks)", 
                placeholder="Ketik ulasan Anda di sini...\n(Contoh: 'Pengemasannya super lelet, saya kecewa berat.')",
                lines=5
            )
            analyze_btn = gr.Button("Analisis Sentimen", variant="primary")
            
        # Kolom Kanan: Output
        with gr.Column(scale=1):
            text_output = gr.Markdown("*(Hasil teks akan tampil di sini)*")
            label_output = gr.Label(label="Distribusi Kepercayaan Prediksi (Confidence Score)")
            
    # Hubungkan tombol dengan function
    analyze_btn.click(
        fn=predict_sentiment, 
        inputs=[text_input], 
        outputs=[text_output, label_output]
    )

if __name__ == "__main__":
    demo.launch()
