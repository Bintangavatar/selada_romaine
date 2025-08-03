import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Klasifikasi Daun Selada Romaine",
    layout="wide",
    page_icon="🥬"
)

# --- CSS Kustom ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0f1117;
    }
    h1, h2, h3, h4, h5, h6, .stMarkdown {
        color: #00FFFF !important;
    }
    .st-emotion-cache-1c7y2kd, .st-emotion-cache-1v0mbdj {
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Fungsi untuk Memuat Model ---
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("97%_traintestval_2kls_mobilenet_augmentasicolab.h5")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None

model = load_model()
class_names = ['Defisiensi-Nutrisi', 'Full-Nutrisi']

# --- Fungsi Prediksi ---
def model_prediction(model, image_file):
    try:
        image = Image.open(image_file).convert('RGB')
        image = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalisasi

        predictions = model.predict(img_array)
        result_index = np.argmax(predictions)
        confidence = np.max(predictions)
        return result_index, confidence
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses gambar: {e}")
        return -1, 0.0

# --- Sidebar Navigasi ---
st.sidebar.title("📋 Navigasi")
menu = st.sidebar.radio("Pilih Halaman", ["🏠 Beranda", "📊 Tentang", "🧪 Sistem Diagnosis"])

# ===============================
# Halaman Beranda
# ===============================
if menu == "🏠 Beranda":
    st.title("🌱 Sistem Klasifikasi Daun Selada Romaine")
    st.image("selada_estetik.jpg", caption="Daun Selada Romaine", use_column_width=True)

    st.markdown("""
    Sistem ini membantu mendeteksi **kesehatan daun selada romaine** secara otomatis menggunakan teknologi *deep learning*.
    
    **Cara kerja sistem:**
    1. Pilih menu **Sistem Diagnosis**.
    2. Unggah gambar daun selada.
    3. Lihat hasil diagnosis dalam hitungan detik.
    
    Prediksi terbagi menjadi dua kelas:
    - 🟢 **Full-Nutrisi** (daun sehat)
    - 🔴 **Defisiensi-Nutrisi** (daun kekurangan unsur hara)

    Mulai dengan memilih halaman **Sistem Diagnosis** dari sidebar!
    """)

# ===============================
# Halaman Tentang
# ===============================
elif menu == "📊 Tentang":
    st.header("📊 Informasi Dataset")
    st.markdown("""
    Dataset dibagi menjadi dua kelas:
    
    - **Full Nutrisi**: 110 citra observasi + 10 dari folder FN (total 120)
    - **Defisiensi Nutrisi**: Gabungan masing-masing 40 citra dari folder N, P, dan K (total 120)
    
    Preprocessing data menggunakan Data Latih dan Validasi, rincian sebagai berikut:
    - **Training** = 70% 
    - **Validasi** = 30% 
    """)

# ===============================
# Halaman Diagnosis
# ===============================
elif menu == "🧪 Sistem Diagnosis":
    st.header("🧪 Sistem Diagnosis Kualitas Daun")
    st.markdown("Pilih metode input gambar:")

    tab1, tab2 = st.tabs(["📁 Unggah Gambar", "📷 Gunakan Kamera"])

    # Tab Unggah File
    with tab1:
        uploaded_file = st.file_uploader("Unggah gambar daun selada romaine:", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # ✅ Validasi ukuran file (maksimal 2MB)
        if uploaded_file.size > 2 * 1024 * 1024:  # 2MB dalam byte
            st.error("❌ Ukuran file terlalu besar! Maksimal 2MB.")
        else:
            st.image(uploaded_file, caption="Gambar yang Diunggah", use_column_width=True)

            if st.button("🔍 Lakukan Prediksi", key="upload_button"):
                if model is None:
                    st.warning("Model belum berhasil dimuat.")
                else:
                    with st.spinner("Menganalisis..."):
                        label_index, confidence = model_prediction(model, uploaded_file)
                        if label_index != -1:
                            st.success(f"✅ Hasil Prediksi: **{class_names[label_index]}**")
                            st.info(f"📈 Tingkat keyakinan: `{confidence:.2%}`")
                        else:
                            st.error("Prediksi gagal dilakukan.")

    # Tab Kamera
    with tab2:
        camera_image = st.camera_input("Ambil gambar daun selada:")
        if camera_image is not None:
            st.image(camera_image, caption="Gambar dari Kamera", use_column_width=True)

            if st.button("🔍 Prediksi dari Kamera", key="camera_button"):
                if model is None:
                    st.warning("Model belum berhasil dimuat.")
                else:
                    with st.spinner("Menganalisis..."):
                        label_index, confidence = model_prediction(model, camera_image)
                        if label_index != -1:
                            st.success(f"✅ Hasil Prediksi: **{class_names[label_index]}**")
                            st.info(f"📈 Tingkat keyakinan: `{confidence:.2%}`")
                        else:
                            st.error("Prediksi gagal dilakukan.")
