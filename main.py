from preprocessing import PreprocessingPipeline
from klasifikasi import SentimentClassifier

if __name__ == "__main__":
    print("="*50)
    print("MEMULAI PIPELINE SENTIMENT ANALYSIS")
    print("="*50)
    
    # 1. Preprocessing
    print("\n[Tahap 1] Preprocessing Data...")
    preprocessor = PreprocessingPipeline(text_col='review_text', target_col='sentiment', slang_dict='slang-indo.csv')
    df_clean = preprocessor.fit_transform('dataset.csv')
    print("Sampel Data Tersimpan:")
    print(df_clean[['sentiment']].head(3))  # review_text diubah menjadi token TF-IDF

    # 2. Setup Machine Learning
    print("\n[Tahap 2] Mempersiapkan Environment Machine Learning...")
    classifier = SentimentClassifier()
    
    # Train Size 80% (Validation 20%)
    classifier.setup(data=df_clean, train_size=0.8)

    # 3. Model Training
    print("\n[Tahap 3] Melatih Model Klasifikasi Sentimen...")
    model_lr = classifier.compare(sort='Accuracy', n_select=3)
    classifier.train("auto")
    
    # 4. Evaluasi (Testing ke Validation Data 20%)
    print("\n[Tahap 4] Mengevaluasi Performa Model...")
    hasil_evaluasi = classifier.score()
    print(hasil_evaluasi)
    
    print("\nPIPELINE SELESAI")
    print("="*50)
