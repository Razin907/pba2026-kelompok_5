import pandas as pd
import re
import warnings
import nltk
from nltk.corpus import stopwords

warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Kelas untuk melakukan pre-processing data sentimen dari Hugging Face.
    Class ini akan memuat file CSV dan melakukan pembersihan teks,
    kasus folding, penanganan slang (bahasa gaul e-commerce), serta
    penghapusan stopword (dengan pengecualian kata negasi).
    
    Catatan: Lemmatization dan Stemming (misal via Sastrawi) dianulir 
    karena memakan waktu komputasi sangat lambat (~2 jam untuk 19k baris) 
    dan slang normalization biasanya sudah cukup untuk model sentimen modern.
    """
    
    def __init__(self, filepath: str, slang_filepath: str = "slang-indo.csv"):
        """
        Inisialisasi DataPreprocessor dengan letak file CSV utama dan file slang.
        
        Args:
            filepath (str): path/lokasi file dataset CSV.
            slang_filepath (str): path file kamus slang CSV (default: slang-indo.csv).
        """
        self.filepath = filepath
        self.data = None
        
        # 1. Kamus Slang (E-commerce / Review context) dinamis dari CSV
        self.slang_dict = {}
        try:
            slang_df = pd.read_csv(slang_filepath)
            # Konversi dataframe ke dictionary: kunci=slang, nilai=formal
            # dropna memastikan tidak ada kata NaN yang masuk ke kamus
            slang_df = slang_df.dropna()
            self.slang_dict = dict(zip(slang_df['slang'], slang_df['formal']))
            print(f"Kamus slang termuat: {len(self.slang_dict)} kata.")
        except Exception as e:
            print(f"File {slang_filepath} tidak ditemukan/gagal dimuat: {e}. Menggunakan kamus slang kosong.")
            
        # 2. Tambahan Slang Manual (Optional, khusus e-commerce)
        self.slang_dict.update({
            "zonk": "jelek", "nyesel": "menyesal", "d": "di", "toped": "tokopedia",
            "murmer": "murah meriah", "rekomen": "rekomendasi", "samsek": "sama sekali",
            "bosenin": "membosankan", "hpx": "hp nya", "dapet": "dapat", 
            "dipake": "dipakai", "kibot": "keyboard", "batok": "adaptor charger", 
            "ky": "seperti", "puol": "banget"
        })
        
        # 2. Persiapan Stopwords
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
            
        # Mengambil stopword bahasa Indonesia
        self.stop_words = set(stopwords.words('indonesian'))
        
        # Pengecualian: Kata-kata negasi tidak boleh dibuang karena sangat penting 
        # untuk menentukan sentimen (contoh: "tidak bagus" vs "bagus")
        negasi = {'tidak', 'bukan', 'jangan', 'belum', 'enggak', 'gak', 'kurang'}
        self.stop_words = self.stop_words - negasi

    def load_data(self) -> pd.DataFrame:
        """
        Membaca dataset dari file CSV menggunakan pandas.
        
        Returns:
            pd.DataFrame: Objek dataframe dari file yang telah di-load.
        """
        try:
            self.data = pd.read_csv(self.filepath)
            print(f"Dataset berhasil dimuat dengan ukuran {self.data.shape}.")
        except Exception as e:
            print(f"Terjadi kesalahan saat membaca file {self.filepath}: {e}")
        return self.data

    def clean_text(self, text: str) -> str:
        """
        Membersihkan teks dari karakter non-alfabet, hashtag, mention, dan link.
        Mengubah menjadi lowercase, memperbaiki kata slang, dan menghapus stopword.
        
        Args:
            text (str): String yang belum dibersihkan (raw text).
            
        Returns:
            str: Teks bersih.
        """
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        # Menghilangkan URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Menghapus punctuation/karakter spesial, hanya membiarkan huruf dan spasi
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Tokenisasi manual per kata
        words = text.split()
        
        cleaned_words = []
        for w in words:
            # 1. Konversi Slang
            w = self.slang_dict.get(w, w)
            # 2. Hapus Stopword
            if w not in self.stop_words:
                cleaned_words.append(w)
                
        # Gabungkan kembali
        text = " ".join(cleaned_words).strip()
        
        return text

    def get_processed_data(self) -> pd.DataFrame:
        """
        Melakukan keseluruhan alur proses (load, clean text pada kolom 'review_text',
        dan drop baris kosong).
        
        Returns:
            pd.DataFrame: Dataframe yang telah di-preprocessing.
        """
        df = self.load_data()
        if df is None:
            return None
            
        print("Memulai proses pembersihan teks...")
        
        # Asumsikan nama kolom adalah 'review_text' dan 'sentiment' melihat sampel dataset.csv
        if 'review_text' in df.columns:
            df['review_text'] = df['review_text'].apply(self.clean_text)
            
            # Hapus baris yang review_text-nya kosong setelah dibersihkan
            df = df[df['review_text'] != ""]
            
            # Pengecekan nilai missing values pada target 'sentiment'
            if 'sentiment' in df.columns:
                df = df.dropna(subset=['sentiment'])
                
            print(f"Dataframe setelah proses cleaning berukuran {df.shape}.")
        else:
            print("Perhatian: kolom 'review_text' tidak ditemukan dalam dataset.")
            
        self.data = df
        return df
