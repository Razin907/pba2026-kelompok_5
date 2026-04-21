import gradio as gr
import pandas as pd
import os
import sys
import torch
from pathlib import Path

# Di Hugging Face Spaces (root script), current_dir = root_dir
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.train_dl import BiLSTMAttention, Vocabulary

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dl_system():
    try:
            
        save_dir = os.path.join(root_dir, "models", "model_dl")
        if not os.path.exists(save_dir):
            save_dir = os.path.join(root_dir, "models")

        vocab_path = os.path.join(save_dir, "vocab_dl.json")
        model_path = os.path.join(save_dir, "best_model.pth")
        
        if not os.path.exists(vocab_path) or not os.path.exists(model_path):
            print(f"File vocab/model DL tidak ditemukan di: {save_dir}")
            return None, None

        # Memuat vocab
        vocab = Vocabulary.load(vocab_path)
        
        # Inisialisasi model
        model = BiLSTMAttention(vocab_size=len(vocab)).to(DEVICE)
        
        # Load bobot / weights model (.pth) pada CPU
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()  
        
        return vocab, model
    except Exception as e:
        print(f"Error memuat DL model: {e}")
        return None, None

vocab, model = load_dl_system()

def predict_sentiment_dl(user_input):
    if model is None or vocab is None:
        return "⚠️ Error: Sistem Deep Learning (.pth) gagal di-load.", None
        
    if not user_input or str(user_input).strip() == "":
        return "⚠️ Mohon ketik ulasan terlebih dahulu.", None
        
    try:
        text_clean = user_input
        # (Preprocessing disederhanakan tanpa library tambahan jika ingin fully pure)
        # 2. Encoding menggunakan Vocabulary (max_len = 128)
        encoded_ids = vocab.encode(text_clean, max_len=128)
        input_tensor = torch.tensor([encoded_ids], dtype=torch.long).to(DEVICE)
        
        # 3. Neural Network Inference
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_label = logits.argmax(dim=1).item()
            
        prob_negatif = float(probs[0])
        prob_positif = float(probs[1])
        
        if pred_label == 1:
            icon = "✅ Ulasan Terindikasi Positif"
            sentiment_text = "Positif"
        else:
            icon = "❌ Ulasan Terindikasi Negatif"
            sentiment_text = "Negatif"
            
        result_text = f"### {icon}\n\nNilai Sentimen PyTorch BiLSTM: **{sentiment_text}**"
        confidence_dict = {
            "Sentimen Positif": prob_positif, 
            "Sentimen Negatif": prob_negatif
        }
        return result_text, confidence_dict
        
    except Exception as e:
        return f"Terjadi kesalahan: {str(e)}", None

with gr.Blocks(title="DL Sentiment Analysis", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 E-Commerce Sentiment Analysis (Deep Learning)")
    gr.Markdown("Deteksi sentimen menggunakan PyTorch BiLSTM + Attention (model `.pth`). App ini sudah siap dideploy di Hugging Face Spaces!")
    
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Masukkan Ulasan Produk", 
                placeholder="'Pengemasannya sangat cepat dan barang bagus, terima kasih!'",
                lines=5
            )
            analyze_btn = gr.Button("Analisis Sentimen (BiLSTM)", variant="primary")
            
        with gr.Column(scale=1):
            text_output = gr.Markdown("*(Hasil teks)*")
            label_output = gr.Label(label="Distribusi Prediksi Softmax")
            
    analyze_btn.click(fn=predict_sentiment_dl, inputs=[text_input], outputs=[text_output, label_output])

if __name__ == "__main__":
    demo.launch()
