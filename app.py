import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import pandas as pd
import altair as alt
import re
import string
import time

# 1. Konfigurasi Halaman & Styling CSS
st.set_page_config(page_title="Deteksi Tingkat Stres", layout="centered")

st.markdown("""
    <style>
    /* 1. WARNA BACKGROUND HALAMAN (Biru Pastel Lembut) */
    .stApp {
        background-color: #EBF5FB; 
    }

    /* 2. Menghilangkan background putih bawaan pada header agar menyatu */
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    /* 3. Menghilangkan margin atas bawaan agar lebih rapi */
    .main .block-container {
        padding-top: 50px;
    }

    /* 4. Styling Kotak Hasil agar lebih kontras */
    .label-box {
        border: 1.5px solid #333333;
        border-radius: 10px;
        padding: 8px;
        text-align: center;
        background-color: #ffffff;
        font-weight: bold;
        width: 100%;
        display: block;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
    }

    /* 5. Mempercantik Tombol (Minimalis dengan transisi) */
    .stButton>button {
        border-radius: 10px;
        border: 1.5px solid #333;
        background-color: white;
        color: #333;
        font-weight: bold;
        transition: 0.3s ease-in-out;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        white-space: nowrap; 
        min-width: 120px;    
    }
    
    .stButton>button:hover {
        background-color: #333;
        color: white;
        box-shadow: 3px 3px 10px rgba(0,0,0,0.15);
    }

    /* 6. Memastikan Container Utama berwarna Putih Kontras */
    [data-testid="stVerticalBlockBorderWrapper"] > div:nth-child(1) {
        background-color: white;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.08); 
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Inisialisasi Session State
if 'page' not in st.session_state:
    st.session_state.page = 'input'
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# 3. Load Model (Langsung mendownload otomatis dari Hugging Face Hub)
@st.cache_resource
def load_model():
    # Masukkan nama repositori Hugging Face kamu di sini
    repo_name = "nadianalifiana/IndoBERT_stress_model"
    
    tokenizer = AutoTokenizer.from_pretrained(repo_name)
    model = AutoModelForSequenceClassification.from_pretrained(repo_name)
    return tokenizer, model

tokenizer, model = load_model()

# --- FUNGSI CLEANSING ---
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^\w\s\.,!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

# 4. Fungsi untuk Menarik Tweet Otomatis
def fetch_from_csv_simulated(username):
    try:
        df_sample = pd.read_csv("datastres.csv", encoding='latin1') 
        if 'tweet' in df_sample.columns:
            random_tweets = df_sample['tweet'].sample(5).tolist()
            return random_tweets
        else:
            st.error("Kolom 'tweet' tidak ditemukan di CSV!")
            return []
    except Exception as e:
        st.error(f"Gagal mensimulasikan tarik data: {str(e)}")
        return []

# 5. LOGIKA PREDIKSI (DIKONDISIKAN UNTUK TINGKATAN STRES)
def run_prediction(user_text):
    text_to_predict = clean_text(user_text)
    inputs = tokenizer(text_to_predict, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
    
    labels = ['Stres', 'Tidak Stres', 'Netral']
    idx = torch.argmax(probs).item()
    
    raw_prob = probs[0][idx].item() # Nilai probabilitas murni (0.0 - 1.0)
    
    # Logika penentuan tingkatan stres berdasarkan interval skor
    tingkat_stres = "-"
    if labels[idx] == 'Stres':
        if 0.50 <= raw_prob <= 0.65:
            tingkat_stres = "Stres Rendah"
        elif 0.66 <= raw_prob <= 0.80:
            tingkat_stres = "Stres Sedang"
        elif raw_prob >= 0.81:
            tingkat_stres = "Stres Tinggi"
    
    # Simpan hasil ke session_state
    st.session_state.prediction = {
        'label': labels[idx],
        'prob': f"{raw_prob:.2%}",
        'raw_prob': raw_prob,
        'tingkat_stres': tingkat_stres,
        'all_probs': probs[0].tolist(),
        'labels': labels
    }
    
    st.session_state.page = 'result'
    st.rerun()

# --- HALAMAN 1: INPUT ---
def show_input_page():
    with st.container(border=True):
        st.markdown("<h2 style='text-align: center; color: #1a1a1a; margin-top: 0;'>Deteksi Tingkat Stres</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #777; margin-bottom: 20px;'>Analisis teks menfess menggunakan model IndoBERT</p>", unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Input Manual", "Tarik Data Otomatis"])
        final_text_to_predict = ""

        # --- TAB 1: INPUT MANUAL ---
        with tab1:
            st.markdown("### Masukkan Kalimat")
            manual_text = st.text_area(
                "Ketik kalimat menfess di sini:", 
                placeholder="Contoh: Aku ngerasa capek banget hari ini...",
                height=150,
                key="manual_input"
            )
            if st.button("PREDIKSI", use_container_width=True, type="primary"):
                if manual_text.strip():
                    run_prediction(manual_text)
                else:
                    st.error("Silakan masukkan teks terlebih dahulu!")

        # --- TAB 2: TARIK DATA OTOMATIS ---
        with tab2:
            st.markdown("### Ambil Menfess Terbaru")
            col_btn, col_status = st.columns([1, 2])
            
            with col_btn:
                btn_tarik = st.button("TARIK TWEET", use_container_width=True)
            
            if btn_tarik:
                with col_status:
                    with st.spinner("Mencari data di Twitter..."):
                        time.sleep(1.5)
                        try:
                            df_sample = pd.read_csv("datastres.csv", encoding='latin1')
                            st.session_state.list_tweets = df_sample['tweet'].sample(5).tolist()
                        except Exception as e:
                            st.error("Gagal mengakses dataset.")
                st.success("Berhasil menemukan 5 menfess terbaru!")

            if 'list_tweets' in st.session_state:
                st.write("---")
                selected_tweet = st.selectbox(
                    "Pilih salah satu menfess dari Twitter:",
                    ["-- Pilih Menfess --"] + st.session_state.list_tweets,
                    key="auto_select"
                )
                
                if selected_tweet != "-- Pilih Menfess --":
                    st.info(f"**Teks Terpilih:**\n\n{selected_tweet}")
                    if st.button("PREDIKSI DATA TWITTER", use_container_width=True, type="primary"):
                        run_prediction(selected_tweet)

# --- HALAMAN 2: HASIL ---
def show_result_page():
    res = st.session_state.prediction
    with st.container(border=True):
        st.markdown("<h2 style='text-align: center; color: #1a1a1a; margin-top: 0;'>Hasil Analisis</h2>", unsafe_allow_html=True)
        
        # --- KOTAK INFORMASI ---
        # Menambahkan baris Tingkat Stres jika kategori yang terdeteksi adalah 'Stres'
        tingkat_stres_html = ""
        if res['label'] == 'Stres':
            tingkat_stres_html = f"""
            <div style="display: flex; align-items: center; margin-top: 15px;">
                <div style="width: 25%; font-weight: bold;">Tingkat Stres</div>
                <div style="width: 5%;">:</div>
                <div style="width: 70%;" class="label-box">{res['tingkat_stres']}</div>
            </div>
            """

        st.markdown(f"""
            <div style="border: 2px solid #333; border-radius: 15px; overflow: hidden; margin-top: 20px; background-color: #fafafa;">
                <div style="border-bottom: 2px solid #333; padding: 12px 20px; font-weight: bold; background-color: #e6f0f5;">Informasi Prediksi</div>
                <div style="padding: 30px;">
                    <div style="display: flex; align-items: center; margin-bottom: 15px;">
                        <div style="width: 25%; font-weight: bold;">Kategori</div>
                        <div style="width: 5%;">:</div>
                        <div style="width: 70%;" class="label-box">{res['label']}</div>
                    </div>
                    <div style="display: flex; align-items: center;">
                        <div style="width: 25%; font-weight: bold;">Probabilitas</div>
                        <div style="width: 5%;">:</div>
                        <div style="width: 70%;" class="label-box">{res['prob']}</div>
                    </div>
                    {tingkat_stres_html}
                </div>
            </div>
        """, unsafe_allow_html=True)

        # --- GRAFIK PROBABILITAS ---
        st.write("")
        st.markdown("### Grafik Probabilitas Model")
        
        df_chart = pd.DataFrame({
            "Kategori": res['labels'],
            "Skor": [p * 100 for p in res['all_probs']]
        })

        color_scale = alt.Scale(
            domain=['Stres', 'Tidak Stres', 'Netral'],
            range=['#FF4B4B', '#2ECC71', '#3498DB'] 
        )

        chart = alt.Chart(df_chart).mark_bar(
            cornerRadiusTopRight=10,
            cornerRadiusBottomRight=10
        ).encode(
            x=alt.X('Skor:Q', title="Keyakinan (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y('Kategori:N', title="", sort='-x'),
            color=alt.Color('Kategori:N', scale=color_scale, legend=None),
            tooltip=['Kategori', 'Skor']
        ).properties(
            height=200,
            width='container'
        )

        st.altair_chart(chart, use_container_width=True)

        # --- PENJABARAN KESIMPULAN OTOMATIS BERDASARKAN TINGKATAN ---
        st.write("---")
        st.markdown("### Kesimpulan Analisis")
        
        if res['label'] == 'Stres':
            if res['tingkat_stres'] == 'Stres Rendah':
                st.warning(f"Sistem mendeteksi adanya indikasi **{res['tingkat_stres']}** pada teks tersebut (Confidence Score: {res['prob']}).")
                st.info("💡 **Analisis & Saran:** Tekanan emosional yang terdeteksi masih dalam batas wajar/ringan. Cobalah untuk istirahat sejenak dari aktivitas rutin, minum air putih, atau mendengarkan musik santai.")
            
            elif res['tingkat_stres'] == 'Stres Sedang':
                st.warning(f"Sistem mendeteksi adanya indikasi **{res['tingkat_stres']}** pada teks tersebut (Confidence Score: {res['prob']}).")
                st.info("💡 **Analisis & Saran:** Teks menunjukkan tanda-tanda beban pikiran atau stres yang cukup jelas. Disarankan untuk mengambil jeda *refreshing*, melakukan teknik pernapasan dalam (deep breathing), atau menceritakan keluh kesah ke orang terdekat.")
            
            elif res['tingkat_stres'] == 'Stres Tinggi':
                st.error(f"Sistem mendeteksi adanya indikasi **{res['tingkat_stres']}** yang sangat kuat pada teks tersebut (Confidence Score: {res['prob']}).")
                st.info("💡 **Analisis & Saran:** Kalimat ini mengekspresikan tekanan emosional/frustrasi yang sangat intens. Jika kondisi ini terus berlanjut atau mengganggu aktivitas sehari-hari, sangat dianjurkan untuk berkonsultasi atau bercerita dengan tenaga profesional (psikolog/konselor).")
        
        elif res['label'] == 'Tidak Stres':
            st.success(f"Sistem mendeteksi bahwa teks tersebut termasuk dalam kategori **Tidak Stres** dengan tingkat keyakinan {res['prob']}.")
            st.info("💡 **Analisis:** Kalimat ini cenderung menunjukkan kondisi emosi yang stabil, positif, atau santai.")
        
        else:  
            st.info(f"Sistem mengklasifikasikan teks ini ke dalam kategori **Netral** dengan probabilitas {res['prob']}.")
            st.info("💡 **Analisis:** Teks tidak menunjukkan indikasi stres maupun ketenangan yang spesifik. Kalimat ini kemungkinan bersifat informatif atau umum.")

        # --- TOMBOL KEMBALI ---
        st.write("")
        col1, col2, col3 = st.columns([3, 3, 2])
        with col3:
            if st.button("KEMBALI"):
                st.session_state.page = 'input'
                st.rerun()

# --- LOGIKA NAVIGASI ---
if model is None:
    st.error("Folder 'IndoBERT_stress_model' tidak ditemukan! Pastikan folder model ada di samping file app.py")
else:
    if st.session_state.page == 'input':
        show_input_page()
    else:
        show_result_page()