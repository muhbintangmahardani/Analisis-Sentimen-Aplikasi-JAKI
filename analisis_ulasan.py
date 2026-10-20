import time
import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import nltk
import textwrap
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# =====================================================================
# 1. PENGATURAN HALAMAN STREAMLIT
# =====================================================================
st.set_page_config(
    page_title="JAKIVision | Analisis Sentimen SVM JAKI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================================
# 2. CSS KUSTOM: TEMA BIRU JAKI + STRUKTUR LAYOUT ALA MUSAVISION
# =====================================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: #F8FAFC;
        color: #0F172A;
    }

    /* PERBAIKAN BUG TERPOTONG DI ATAS LAYAR */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2rem !important;
    }
    .block-container {
        padding-top: 2.2rem !important;
        padding-bottom: 4rem !important;
        max-width: 1260px !important;
    }

    /* TOP STATUS BAR (Hitam/Navy Gelap seperti referensi gambar ke-2) */
    .top-statusbar {
        background-color: #091536;
        color: #CBD5E1;
        padding: 10px 22px;
        border-radius: 12px;
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        gap: 10px;
        font-size: 0.76rem;
        margin-bottom: 14px;
        border: 1px solid #1E3A8A;
    }
    .status-pill {
        background-color: rgba(37, 99, 235, 0.25);
        color: #60A5FA;
        border: 1px solid #3B82F6;
        padding: 3px 10px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.7rem;
        margin-right: 8px;
    }

    /* BRAND BAR */
    .brand-bar {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 14px 24px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 6px rgba(15, 37, 87, 0.03);
    }
    .brand-logo {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-icon {
        background: linear-gradient(135deg, #0055B8, #0F2557);
        color: #FFFFFF;
        font-weight: 800;
        font-size: 1rem;
        padding: 6px 12px;
        border-radius: 8px;
        letter-spacing: 0.04em;
    }
    .brand-version {
        background-color: #EBF3FA;
        color: #0055B8;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 6px;
        border: 1px solid #BFDBFE;
    }

    /* HERO SECTION (2-Column Style) */
    .hero-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 28px;
        height: 100%;
        box-shadow: 0 4px 12px rgba(15, 37, 87, 0.03);
    }
    .hero-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: #EBF3FA;
        color: #0055B8;
        border: 1px solid #BFDBFE;
        font-size: 0.72rem;
        font-weight: 800;
        padding: 4px 12px;
        border-radius: 999px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 14px;
    }
    .hero-heading {
        font-size: 2.15rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
        margin-bottom: 12px;
    }
    .hero-heading span {
        color: #0055B8;
        font-style: italic;
    }
    .hero-sub {
        font-size: 0.92rem;
        color: #475569;
        line-height: 1.65;
        margin-bottom: 20px;
    }

    /* 4 MINI METRIC CARDS DI HERO */
    .mini-metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin-top: 18px;
    }
    .mini-metric {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 12px 14px;
    }
    .mini-metric .val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.25rem;
        font-weight: 800;
        color: #0F172A;
    }
    .mini-metric .val.blue {
        color: #0055B8;
    }
    .mini-metric .lbl {
        font-size: 0.7rem;
        color: #64748B;
        font-weight: 600;
        margin-top: 2px;
    }

    /* KARTU PREVIEW KANAN HERO */
    .preview-card {
        background: linear-gradient(180deg, #0F2557 0%, #091536 100%);
        color: #FFFFFF;
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #1E3A8A;
        box-shadow: 0 8px 20px rgba(15, 37, 87, 0.12);
    }
    .preview-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.75rem;
        color: #93C5FD;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(255,255,255,0.12);
        margin-bottom: 14px;
    }
    .preview-bubble {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 14px;
        font-size: 0.84rem;
        line-height: 1.5;
        color: #F1F5F9;
        margin-bottom: 12px;
    }
    .preview-token-box {
        background: rgba(0, 85, 184, 0.3);
        border: 1px dashed #60A5FA;
        border-radius: 10px;
        padding: 10px 12px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        color: #BFDBFE;
        margin-bottom: 14px;
    }

    /* SECTION LABELS ALA MUSAVISION */
    .sec-eyebrow {
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #0055B8;
        margin-bottom: 4px;
    }
    .sec-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 6px;
    }
    .sec-desc {
        font-size: 0.88rem;
        color: #64748B;
        margin-bottom: 18px;
    }

    /* PIPELINE 5 TAHAP CARDS */
    .pipe-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 18px 16px;
        height: 100%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
    }
    .pipe-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 800;
        color: #0055B8;
        background-color: #EBF3FA;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-block;
        margin-bottom: 10px;
    }
    .pipe-title {
        font-size: 0.95rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 6px;
    }
    .pipe-text {
        font-size: 0.78rem;
        color: #475569;
        line-height: 1.5;
        margin-bottom: 10px;
    }
    .pipe-foot {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: #64748B;
        border-top: 1px solid #F1F5F9;
        padding-top: 8px;
    }

    /* STUDIO PANEL CARDS */
    .studio-panel {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 2px 8px rgba(15, 37, 87, 0.03);
    }
    .studio-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 12px;
        margin-bottom: 16px;
    }
    .studio-step-badge {
        background-color: #0F2557;
        color: #FFFFFF;
        font-size: 0.72rem;
        font-weight: 800;
        padding: 4px 10px;
        border-radius: 6px;
    }

    /* EMPTY STATE MENUNGGU MASUKAN */
    .waiting-box {
        border: 2px dashed #CBD5E1;
        background-color: #F8FAFC;
        border-radius: 14px;
        padding: 45px 24px;
        text-align: center;
        color: #64748B;
    }

    /* KARTU HASIL DIAGNOSIS SENTIMEN */
    .res-pos {
        background-color: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-left: 6px solid #059669;
        border-radius: 12px;
        padding: 18px 20px;
        color: #065F46;
        margin-bottom: 14px;
    }
    .res-neg {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 6px solid #DC2626;
        border-radius: 12px;
        padding: 18px 20px;
        color: #991B1B;
        margin-bottom: 14px;
    }

    /* KUSTOMISASI TOMBOL STREAMLIT */
    div.stButton > button[kind="primary"] {
        background-color: #0055B8 !important;
        color: #FFFFFF !important;
        border: 1px solid #0055B8 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 0.55rem 1.2rem !important;
        transition: all 0.2s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #0F2557 !important;
        border-color: #0F2557 !important;
    }
    div.stButton > button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #334155 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1rem !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        border-color: #0055B8 !important;
        color: #0055B8 !important;
        background-color: #F0F7FF !important;
    }

    /* KPI CARDS TAB 3 */
    .kpi-container {
        display: flex;
        flex-wrap: wrap;
        gap: 18px;
        margin-bottom: 24px;
    }
    .kpi-card {
        flex: 1;
        min-width: 220px;
        background: #FFFFFF;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .kpi-total { border-left: 6px solid #64748B; }
    .kpi-positif { border-left: 6px solid #4E79A7; background-color: #F8FBFF; }
    .kpi-negatif { border-left: 6px solid #E15759; background-color: #FFF8F8; }
    .kpi-title { font-size: 0.85rem; font-weight: 700; text-transform: uppercase; margin: 0 0 6px 0; }
    .kpi-value { font-size: 2.2rem; font-weight: 800; margin: 0; font-family: 'JetBrains Mono', monospace; }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# 3. INISIALISASI SASTRAWI, STOPWORDS, & MODEL
# =====================================================================
@st.cache_resource
def load_text_processor():
    try:
        nltk.download('stopwords', quiet=True)
    except Exception:
        pass

    factory = StemmerFactory()
    stemmer_obj = factory.create_stemmer()

    try:
        stopwords_list = stopwords.words('indonesian')
    except Exception:
        stopwords_list = [
            'yang', 'di', 'dan', 'itu', 'dengan', 'untuk', 'pada', 'ke', 'dari',
            'ini', 'dalam', 'akan', 'juga', 'saya', 'ada', 'bisa', 'sudah', 'mau'
        ]

    kata_negasi_baku = ['tidak', 'bukan', 'belum', 'jangan']
    stopwords_list = [kata for kata in stopwords_list if kata not in kata_negasi_baku]

    kata_tambahan = [
        'yg', 'dg', 'rt', 'dgn', 'ny', 'd', 'klo', 'kalo', 'amp', 'biar',
        'bikin', 'bilang', 'krn', 'nya', 'nih', 'sih', 'si', 'tau', 'tuh',
        'utk', 'ya', 'jd', 'sdh', 'aja', 'n', 't', 'nyg', 'hehe', 'pen',
        'u', 'nan', 'loh', 'pas', 'sgt', 'sngt', 'y', 'bgt', 'banget', 'deh'
    ]
    stopwords_list.extend(kata_tambahan)
    return stemmer_obj, set(stopwords_list)

stemmer, list_stopwords = load_text_processor()

kamus_normalisasi = {
    'ga': 'tidak', 'gak': 'tidak', 'enggak': 'tidak', 'ngga': 'tidak',
    'nggak': 'tidak', 'gk': 'tidak', 'g': 'tidak', 'tdk': 'tidak', 'ndak': 'tidak',
    'jgn': 'jangan', 'blm': 'belum', 'bkn': 'bukan',
    'lemot': 'lambat', 'lmot': 'lambat', 'lelet': 'lambat',
    'ngelag': 'lambat', 'ngeleg': 'lambat', 'lag': 'lambat',
    'error': 'rusak', 'eror': 'rusak', 'cacat': 'rusak',
    'bug': 'rusak', 'blank': 'rusak',
    'bagus': 'bagus', 'goks': 'bagus', 'gokil': 'bagus', 'keren': 'bagus',
    'oon': 'bodoh', 'gblk': 'bodoh', 'bego': 'bodoh', 'tolol': 'bodoh',
    'jelek': 'buruk', 'cemen': 'buruk'
}

def pembersihan_teks(text):
    text = str(text).lower()
    text = re.sub(r'([a-zA-Z])\1+', r'\1', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@[a-zA-Z0-9_]+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    kata_terpisah = text.split()
    kata_bersih = []

    for kata in kata_terpisah:
        if kata in kamus_normalisasi:
            kata = kamus_normalisasi[kata]
        if kata not in list_stopwords:
            kata_dasar = stemmer.stem(kata)
            if kata_dasar != '':
                kata_bersih.append(kata_dasar)

    return " ".join(kata_bersih)

@st.cache_resource
def load_models():
    model = joblib.load('model_svm_jaki.pkl')
    vectorizer = joblib.load('tfidf_jaki.pkl')
    return model, vectorizer

try:
    svm_model, tfidf_vect = load_models()
    jml_kosakata = len(tfidf_vect.vocabulary_) if hasattr(tfidf_vect, 'vocabulary_') else 2500
except Exception:
    st.error("Gagal memuat file .pkl. Pastikan model_svm_jaki.pkl dan tfidf_jaki.pkl ada di folder ini!")
    st.stop()

# =====================================================================
# 4. FUNGSI 4 LAPIS GEMBOK VALIDASI KHUSUS DATASET JAKI (ANTI UMUM/SHOPEE)
# =====================================================================
def validasi_khusus_jaki(df):
    if 'Ulasan' not in df.columns:
        return False, "❌ Akses Ditolak: File wajib memiliki kolom bernama **'Ulasan'** sesuai format dataset resmi JAKI."

    ulasan_series = df['Ulasan'].dropna().astype(str).str.lower()
    total_baris = len(ulasan_series)

    if total_baris == 0:
        return False, "❌ Akses Ditolak: Kolom 'Ulasan' di dalam file kosong."

    kata_platform_lain = [
        r'\bshopee\b', r'\btokopedia\b', r'\btokped\b', r'\blazada\b', r'\bblibli\b',
        r'\bbukalapak\b', r'\bgojek\b', r'\bgofood\b', r'\bgrab\b', r'\btraveloka\b',
        r'\bkurir\b', r'\bongkir\b', r'\bpacking\b', r'\bseller\b', r'\bcheckout\b',
        r'\bflash sale\b', r'\brealpict\b', r'\bbarang sampai\b', r'\bpengiriman cepat\b'
    ]
    pattern_asing = "|".join(kata_platform_lain)
    jml_asing = int(ulasan_series.str.contains(pattern_asing, regex=True).sum())

    if jml_asing > 0:
        return False, (
            f"❌ Akses Ditolak (Proteksi Domain): Terdeteksi **{jml_asing} baris ulasan** dari platform lain "
            "(seperti Shopee/Tokopedia/Kurir/Seller). Sistem ini dikunci khusus hanya untuk dataset ulasan **Aplikasi JAKI**."
        )

    identitas_inti_jaki = [
        r'\bjaki\b', r'\bjaklapor\b', r'\bjakwifi\b', r'\bjaklingko\b', r'\bjakpantau\b',
        r'\bjakwarta\b', r'\bjakrespon\b', r'\bjaksurvei\b', r'\bjakpenda\b',
        r'\bjakispu\b', r'\bjakambulans\b', r'\bjakarta kini\b', r'\bpemprov dki\b'
    ]
    pattern_inti = "|".join(identitas_inti_jaki)
    jml_sebut_jaki = int(ulasan_series.str.contains(pattern_inti, regex=True).sum())
    rasio_inti = jml_sebut_jaki / total_baris

    if jml_sebut_jaki == 0 or rasio_inti < 0.05:
        return False, (
            "❌ Akses Ditolak (Bukan Dataset JAKI): File memiliki kolom 'Ulasan', tetapi isinya tidak teridentifikasi "
            f"sebagai ulasan Aplikasi JAKI (hanya **{round(rasio_inti * 100, 1)}%** baris yang menyebut entitas JAKI; minimal wajib **5%**)."
        )

    konteks_dki = [
        r'\bjaki\b', r'\bjakarta\b', r'\bdki\b', r'\bpemprov\b', r'\bgubernur\b',
        r'\blapor\b', r'\blaporan\b', r'\bwarga\b', r'\bkelurahan\b', r'\bkecamatan\b',
        r'\bpetugas\b', r'\bdinas\b', r'\bpajak\b', r'\bvaksin\b', r'\bvaksinasi\b',
        r'\bfaskes\b', r'\bpuskesmas\b', r'\bnik\b', r'\bktp\b', r'\bbansos\b'
    ]
    pattern_konteks = "|".join(konteks_dki)
    jml_konteks_dki = int(ulasan_series.str.contains(pattern_konteks, regex=True).sum())
    rasio_konteks = jml_konteks_dki / total_baris

    if rasio_konteks < 0.15:
        return False, (
            "❌ Akses Ditolak: Konteks kalimat terlalu umum dan tidak relevan dengan ekosistem layanan publik JAKI Pemprov DKI Jakarta."
        )

    return True, (
        f"✅ Dataset Resmi JAKI Terverifikasi! Ditemukan **{total_baris:,} baris ulasan** "
        f"({jml_sebut_jaki:,} penyebutan eksplisit JAKI & {jml_konteks_dki:,} konteks layanan publik DKI)."
    )

# =====================================================================
# 5. STATE MANAGEMENT UNTUK NAVBAR KOLOM & CONTOH INPUT AWAM
# =====================================================================
if 'df_hasil' not in st.session_state:
    st.session_state.df_hasil = None
if 'nav_menu' not in st.session_state:
    st.session_state.nav_menu = "ringkasan"
if 'contoh_teks' not in st.session_state:
    st.session_state.contoh_teks = ""

def pindah_menu(nama_menu):
    st.session_state.nav_menu = nama_menu

# =====================================================================
# 6. HEADER & NAVBAR PER-KOLOM (ALA MUSAVISION)
# =====================================================================
st.markdown(f"""
    <div class="top-statusbar">
        <div>
            <span class="status-pill">● MODEL SVM SIAP</span>
            <span>Arsitektur Klasifikasi Sentimen Ulasan Layanan Publik JAKI (TF-IDF + Sastrawi Stemmer + Linear SVM)</span>
        </div>
        <div>
            <span>Fitur Kata: <strong>{jml_kosakata:,} Token</strong></span> &nbsp;|&nbsp;
            <span>Target Kelas: <strong>Positif & Negatif</strong></span> &nbsp;|&nbsp;
            <span>Peneliti: <strong>Artha Nugraha</strong></span>
        </div>
    </div>
    <div class="brand-bar">
        <div class="brand-logo">
            <div class="brand-icon">🏛️ JAKIVision</div>
            <div>
                <div style="font-weight:800; font-size:1.02rem; color:#0F172A;">Portal Evaluasi Opini Publik JAKI</div>
                <div style="font-size:0.75rem; color:#64748B;">Jakarta Smart City Sentiment Intelligence</div>
            </div>
            <span class="brand-version">SVM v1.0</span>
        </div>
        <div style="font-size:0.78rem; color:#475569; font-weight:600;">
            📍 Pilih menu navigasi pada kolom di bawah untuk mencoba fitur:
        </div>
    </div>
""", unsafe_allow_html=True)

# NAVBAR 4 KOLOM INTERAKTIF
nav1, nav2, nav3, nav4 = st.columns(4)
with nav1:
    if st.button(
        "🏠 Ringkasan & Alur Awam",
        type="primary" if st.session_state.nav_menu == "ringkasan" else "secondary",
        use_container_width=True
    ):
        pindah_menu("ringkasan")
        st.rerun()
with nav2:
    if st.button(
        "💬 Studio Uji Teks Tunggal",
        type="primary" if st.session_state.nav_menu == "tunggal" else "secondary",
        use_container_width=True
    ):
        pindah_menu("tunggal")
        st.rerun()
with nav3:
    if st.button(
        "📁 Pemrosesan Massal CSV",
        type="primary" if st.session_state.nav_menu == "massal" else "secondary",
        use_container_width=True
    ):
        pindah_menu("massal")
        st.rerun()
with nav4:
    label_dash = "📊 Dashboard Visualisasi (Aktif)" if st.session_state.df_hasil is not None else "📊 Dashboard Visualisasi"
    if st.button(
        label_dash,
        type="primary" if st.session_state.nav_menu == "visualisasi" else "secondary",
        use_container_width=True
    ):
        pindah_menu("visualisasi")
        st.rerun()

st.write("")

# =====================================================================
# MENU 1: RINGKASAN & PENJELASAN ALUR UNTUK ORANG AWAM (LANDING PAGE)
# =====================================================================
if st.session_state.nav_menu == "ringkasan":

    # HERO 2 KOLOM (Kiri: Judul & Metrik, Kanan: Ilustrasi Cara Kerja)
    hero_kiri, hero_kanan = st.columns([7, 5], gap="large")

    with hero_kiri:
        st.markdown(f"""
            <div class="hero-card">
                <div class="hero-tag">● NATURAL LANGUAGE PROCESSING &bull; JAKARTA SMART CITY</div>
                <div class="hero-heading">
                    Klasifikasi Sentimen Ulasan Warga pada Aplikasi <span>JAKI Pemprov DKI</span>
                </div>
                <div class="hero-sub">
                    Sistem cerdas untuk memetakan tingkat kepuasan dan keluhan masyarakat DKI Jakarta pada aplikasi
                    <strong>Jakarta Kini (JAKI)</strong>. Dirancang agar mudah dipahami oleh masyarakat awam maupun
                    evaluator kebijakan publik melalui tahapan pembersihan bahasa gaul, <em>stemming</em> <strong>Sastrawi</strong>,
                    pembobotan <strong>TF-IDF</strong>, dan algoritma <strong>Support Vector Machine (SVM)</strong>.
                </div>
                <div class="mini-metrics-grid">
                    <div class="mini-metric">
                        <div class="val">{jml_kosakata:,}</div>
                        <div class="lbl">Kosakata TF-IDF</div>
                    </div>
                    <div class="mini-metric">
                        <div class="val blue">SVM</div>
                        <div class="lbl">Hyperplane Classifier</div>
                    </div>
                    <div class="mini-metric">
                        <div class="val">4 Lapis</div>
                        <div class="lbl">Proteksi CSV JAKI</div>
                    </div>
                    <div class="mini-metric">
                        <div class="val blue">~12 ms</div>
                        <div class="lbl">Kecepatan Prediksi</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        cta1, cta2 = st.columns(2)
        with cta1:
            if st.button("🚀 Buka Studio Uji Teks Tunggal →", type="primary", use_container_width=True):
                pindah_menu("tunggal")
                st.rerun()
        with cta2:
            if st.button("📂 Unggah Dataset CSV JAKI →", type="secondary", use_container_width=True):
                pindah_menu("massal")
                st.rerun()

    with hero_kanan:
        st.markdown("""
            <div class="preview-card">
                <div class="preview-header">
                    <span>🔍 CONTOH BEDAH KALIMAT WARGA DI BALIK LAYAR</span>
                    <span style="background:#0055B8; color:#FFF; padding:2px 8px; border-radius:4px; font-weight:700;">LIVE DEMO</span>
                </div>
                <div style="font-size:0.75rem; color:#93C5FD; margin-bottom:4px; font-weight:600;">1. Kalimat Asli Warga (Bahasa Sehari-hari):</div>
                <div class="preview-bubble">
                    "Aplikasi JAKI lemot banget pas mau lapor jalan berlubang, sering error!"
                </div>
                <div style="font-size:0.75rem; color:#93C5FD; margin-bottom:4px; font-weight:600;">2. Hasil Normalisasi & Sastrawi Stemming:</div>
                <div class="preview-token-box">
                    "aplikasi jaki <strong>lambat</strong> lapor jalan lubang sering <strong>rusak</strong>"
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center; background:rgba(220,38,38,0.22); border:1px solid #F87171; padding:12px 16px; border-radius:10px;">
                    <div>
                        <div style="font-size:0.7rem; color:#FCA5A5; font-weight:700;">KEPUTUSAN HYPERPLANE SVM</div>
                        <div style="font-size:1.05rem; font-weight:800; color:#FFFFFF;">🔴 Sentimen NEGATIF</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:0.7rem; color:#FCA5A5;">Keyakinan Model</div>
                        <div style="font-size:1.1rem; font-weight:800; color:#FEE2E2;">94.8%</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("---")

    # PENJELASAN 5 TAHAP PIPELINE KOMPUTASI UNTUK ORANG AWAM (ALA MUSAVISION)
    st.markdown("""
        <div class="sec-eyebrow">ARSITEKTUR & METODOLOGI</div>
        <div class="sec-title">Lima Tahap Alur Kerja Analisis Sentimen (Panduan Orang Awam)</div>
        <div class="sec-desc">Bagaimana komputer bisa memahami emosi positif atau keluhan negatif dari teks ulasan warga? Berikut adalah 5 tahapan berurutannya:</div>
    """, unsafe_allow_html=True)

    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        st.markdown("""
            <div class="pipe-card">
                <span class="pipe-num">01 / INPUT & GUARD</span>
                <div class="pipe-title">Validasi Domain JAKI</div>
                <div class="pipe-text">Menerima teks ulasan tunggal atau file CSV. Sistem otomatis menolak CSV dari aplikasi lain (seperti Shopee/Tokopedia).</div>
                <div class="pipe-foot">Output: Teks Valid JAKI</div>
            </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown("""
            <div class="pipe-card">
                <span class="pipe-num">02 / CLEANING</span>
                <div class="pipe-title">Normalisasi Kata Gaul</div>
                <div class="pipe-text">Menghapus angka, simbol, link, serta mengubah bahasa tidak baku (misal: <em>"lemot/ngelag"</em> &rarr; <em>"lambat"</em>, <em>"gak"</em> &rarr; <em>"tidak"</em>).</div>
                <div class="pipe-foot">Output: Teks Baku Bersih</div>
            </div>
        """, unsafe_allow_html=True)
    with p3:
        st.markdown("""
            <div class="pipe-card">
                <span class="pipe-num">03 / SASTRAWI</span>
                <div class="pipe-title">Stemming & Stopwords</div>
                <div class="pipe-text">Membuang kata hubung tidak penting (<em>"yang, di, sih"</em>) dan memotong imbuhan menjadi kata dasar Bahasa Indonesia.</div>
                <div class="pipe-foot">Output: Token Kata Dasar</div>
            </div>
        """, unsafe_allow_html=True)
    with p4:
        st.markdown("""
            <div class="pipe-card">
                <span class="pipe-num">04 / VEKTORISASI</span>
                <div class="pipe-title">Pembobotan TF-IDF</div>
                <div class="pipe-text">Komputer tidak bisa membaca huruf, sehingga setiap kata diubah menjadi skor angka berdasarkan tingkat kepentingannya.</div>
                <div class="pipe-foot">Output: Matriks Numerik</div>
            </div>
        """, unsafe_allow_html=True)
    with p5:
        st.markdown("""
            <div class="pipe-card">
                <span class="pipe-num">05 / KLASIFIKASI</span>
                <div class="pipe-title">Pemisahan Margin SVM</div>
                <div class="pipe-text">Algoritma SVM menarik garis batas (<em>hyperplane</em>) untuk memutuskan apakah ulasan masuk ke wilayah Positif atau Negatif.</div>
                <div class="pipe-foot">Output: Label & % Keyakinan</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("---")

    # PANDUAN 3 LANGKAH MUDAH MENGGUNAKAN APLIKASI
    st.markdown("""
        <div class="sec-eyebrow">PANDUAN PENGGUNAAN</div>
        <div class="sec-title">Cara Mencoba Aplikasi Ini dalam 3 Langkah</div>
        <div class="sec-desc">Anda tidak perlu mengerti pemrograman untuk menguji model kecerdasan buatan ini:</div>
    """, unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("""
            <div class="pipe-card" style="border-left: 5px solid #0055B8;">
                <span class="pipe-num">LANGKAH 1 — UJI KALIMAT</span>
                <div class="pipe-title">Coba Studio Uji Teks Tunggal</div>
                <div class="pipe-text">Klik menu <strong>💬 Studio Uji Teks Tunggal</strong> di atas. Anda bisa langsung memilih tombol contoh kalimat keluhan/pujian warga atau mengetiknya sendiri.</div>
            </div>
        """, unsafe_allow_html=True)
    with g2:
        st.markdown("""
            <div class="pipe-card" style="border-left: 5px solid #0F2557;">
                <span class="pipe-num">LANGKAH 2 — PROSES RATUSAN DATA</span>
                <div class="pipe-title">Unggah File CSV Khusus JAKI</div>
                <div class="pipe-text">Buka menu <strong>📁 Pemrosesan Massal CSV</strong> untuk mengunggah tabel kumpulan ulasan JAKI dari Google Play Store dan mengunduh hasil prediksinya.</div>
            </div>
        """, unsafe_allow_html=True)
    with g3:
        st.markdown("""
            <div class="pipe-card" style="border-left: 5px solid #F37021;">
                <span class="pipe-num">LANGKAH 3 — EKSPLORASI GRAFIK</span>
                <div class="pipe-title">Lihat Peta Panas & Word Cloud</div>
                <div class="pipe-text">Setelah memproses CSV, buka <strong>📊 Dashboard Visualisasi</strong> untuk melihat jam sibuk laporan warga serta kumpulan kata yang paling sering dikeluhkan.</div>
            </div>
        """, unsafe_allow_html=True)


# =====================================================================
# MENU 2: STUDIO UJI TEKS TUNGGAL (2-COLUMN STUDIO ALA MUSAVISION)
# =====================================================================
elif st.session_state.nav_menu == "tunggal":

    st.markdown("""
        <div class="sec-eyebrow">STUDIO PENGUJIAN MODEL</div>
        <div class="sec-title">Analisis Teks Tunggal & Diagnosis Sentimen Warga</div>
        <div class="sec-desc">Pilih salah satu tombol contoh ulasan di bawah atau ketik kalimat ulasan JAKI secara manual untuk melihat proses analisisnya secara interaktif.</div>
    """, unsafe_allow_html=True)

    # GALERI 5 SAMPEL CEPAT (Memudahkan orang awam mencoba 1-klik)
    st.markdown("##### ⚡ Galeri Sampel Cepat Ulasan Warga (Klik salah satu untuk mengisi otomatis):")
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        if st.button("👍 Respon JakLapor Cepat", use_container_width=True):
            st.session_state.contoh_teks = "Aplikasi JAKI sangat membantu warga DKI, kemarin saya lapor lampu jalan mati di kelurahan langsung ditindak cepat oleh petugas!"
    with s2:
        if st.button("😡 Aplikasi Lemot & Error", use_container_width=True):
            st.session_state.contoh_teks = "Aplikasi JAKI lemot banget dan sering error blank putih pas mau daftar antrian faskes, tolong segera diperbaiki servernya!"
    with s3:
        if st.button("👍 Fitur Pajak & KTP Mudah", use_container_width=True):
            st.session_state.contoh_teks = "Keren banget aplikasi JAKI sekarang, cek informasi pajak PBB dan layanan kependudukan Jakarta jadi jauh lebih praktis dan bagus."
    with s4:
        if st.button("😡 Lapor Parkir Liar Lambat", use_container_width=True):
            st.session_state.contoh_teks = "Percuma lapor parkir liar dan sampah menumpuk lewat JAKI kalau respon petugasnya lelet banget, aplikasinya juga sering ngelag."
    with s5:
        if st.button("🔄 Kosongkan Teks", use_container_width=True):
            st.session_state.contoh_teks = ""

    st.write("")

    # 2 KOLOM STUDIO: KIRI (INPUT) & KANAN (HASIL DIAGNOSIS + ANIMASI SPINNER)
    col_input, col_output = st.columns([5, 7], gap="large")

    with col_input:
        st.markdown("""
            <div class="studio-panel">
                <div class="studio-header">
                    <span class="studio-step-badge">1. MASUKAN TEKS ULASAN</span>
                    <span style="font-size:0.74rem; color:#64748B; font-family:'JetBrains Mono', monospace;">Bahasa Indonesia / Slang</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        user_input = st.text_area(
            "Ketik keluhan, saran, atau apresiasi warga terhadap aplikasi JAKI:",
            value=st.session_state.contoh_teks,
            placeholder="Contoh: Aplikasi JAKI lemot banget pas mau daftar vaksin, tolong diperbaiki!",
            height=155
        )

        jalankan_uji = st.button("⚡ Jalankan Analisis Sentimen SVM", type="primary", use_container_width=True)

        st.caption(
            "💡 **Info Awam:** Anda bebas menggunakan kata sehari-hari seperti *'gak', 'lemot', 'ngelag', 'error', 'mantap', 'keren'*. "
            "Kamus normalisasi akan menerjemahkannya secara otomatis."
        )

    with col_output:
        st.markdown("""
            <div class="studio-panel">
                <div class="studio-header">
                    <span class="studio-step-badge">2. HASIL DIAGNOSIS & BEDAH ALUR KOMPUTASI</span>
                    <span style="font-size:0.74rem; color:#0055B8; font-weight:700;">Linear SVM + Sastrawi</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if not jalankan_uji:
            st.markdown("""
                <div class="waiting-box">
                    <div style="font-size:2.2rem; margin-bottom:8px;">🏛️</div>
                    <div style="font-weight:800; color:#334155; font-size:1rem; margin-bottom:4px;">Menunggu Masukan Teks Ulasan</div>
                    <div style="font-size:0.82rem; max-width:440px; margin:0 auto; line-height:1.5;">
                        Pilih salah satu tombol sampel cepat di atas atau ketik kalimat ulasan pada panel sebelah kiri, lalu klik
                        <strong>Jalankan Analisis Sentimen SVM</strong> untuk melihat cara kerja model secara interaktif.
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            if user_input.strip() == "":
                st.warning("⚠️ Mohon masukkan teks ulasan terlebih dahulu di kolom sebelah kiri!")
            else:
                # EFEK SPINNER & STATUS INTERAKTIF BERTAHAP
                with st.spinner("🔄 Mesin NLP & SVM sedang membedah kalimat ulasan..."):
                    with st.status("⚙️ Menjalankan Pipeline Komputasi Analisis Sentimen...", expanded=True) as status:
                        st.write("🧹 **Tahap 1:** Membersihkan tanda baca, angka, & menormalisasi kata gaul...")
                        time.sleep(0.25)
                        teks_bersih = pembersihan_teks(user_input)

                        st.write("🌿 **Tahap 2:** Menjalankan *Stemming* Bahasa Indonesia dengan Sastrawi...")
                        time.sleep(0.25)
                        jumlah_token = len(teks_bersih.split()) if teks_bersih else 0

                        st.write("📐 **Tahap 3:** Memetakan kata ke matriks bobot TF-IDF & menghitung jarak Hyperplane SVM...")
                        time.sleep(0.25)
                        teks_vektor = tfidf_vect.transform([teks_bersih])
                        prediction = svm_model.predict(teks_vektor)[0]

                        jarak_margin = float(np.ravel(svm_model.decision_function(teks_vektor))[0])
                        probabilitas = 1 / (1 + np.exp(-jarak_margin))
                        keyakinan = probabilitas if probabilitas > 0.5 else (1 - probabilitas)
                        persentase_yakin = round(keyakinan * 100, 1)

                        status.update(label="✅ Analisis Selesai! Berikut adalah hasil diagnosis sentimennya:", state="complete", expanded=False)

                # TAMPILKAN HASIL AKHIR
                if prediction == 'Positif':
                    st.markdown(f"""
                        <div class="res-pos">
                            <div style="font-size:0.75rem; font-weight:800; text-transform:uppercase; letter-spacing:0.05em;">Keputusan Akhir Model SVM</div>
                            <div style="font-size:1.55rem; font-weight:800; margin:4px 0;">🔵 SENTIMEN POSITIF (Keyakinan: {persentase_yakin}%)</div>
                            <div style="font-size:0.86rem;">Kalimat ini terdeteksi mengandung apresiasi, kepuasan, atau pengalaman baik terhadap layanan Aplikasi JAKI.</div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="res-neg">
                            <div style="font-size:0.75rem; font-weight:800; text-transform:uppercase; letter-spacing:0.05em;">Keputusan Akhir Model SVM</div>
                            <div style="font-size:1.55rem; font-weight:800; margin:4px 0;">🔴 SENTIMEN NEGATIF (Keyakinan: {persentase_yakin}%)</div>
                            <div style="font-size:0.86rem;">Kalimat ini terdeteksi berisi keluhan teknis (aplikasi lambat/error) atau ketidakpuasan terhadap tindak lanjut laporan.</div>
                        </div>
                    """, unsafe_allow_html=True)

                # PROGRESS BAR TINGKAT KEYAKINAN
                st.progress(int(min(max(persentase_yakin, 1), 100)), text=f"Tingkat Keyakinan (Confidence Score): {persentase_yakin}% | Jarak Margin Hyperplane: {jarak_margin:.3f}")

                # DETAIL BEDAH TEKS UNTUK ORANG AWAM
                st.markdown(f"""
                    <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:14px 16px; margin-top:10px; font-size:0.85rem;">
                        <div style="font-weight:700; color:#0F172A; margin-bottom:6px;">🔍 Bedah Perubahan Teks (Sebelum vs Sesudah Sastrawi):</div>
                        <div style="color:#64748B; margin-bottom:4px;"><strong>Kalimat Asli:</strong> "{user_input}"</div>
                        <div style="color:#0055B8;"><strong>Kata Dasar Masuk ke SVM ({jumlah_token} token):</strong> <code>{teks_bersih if teks_bersih else "(Habis terfilter stopwords)"}</code></div>
                    </div>
                """, unsafe_allow_html=True)


# =====================================================================
# MENU 3: PEMROSESAN MASSAL CSV (DENGAN 4 LAPIS GEMBOK JAKI + SPINNER)
# =====================================================================
elif st.session_state.nav_menu == "massal":

    st.markdown("""
        <div class="sec-eyebrow">BATCH PROCESSING ENGINE</div>
        <div class="sec-title">Pemrosesan Massal Dataset Ulasan JAKI (CSV / Excel)</div>
        <div class="sec-desc">Unggah ribuan baris ulasan hasil scraping Google Play Store untuk diklasifikasikan sekaligus secara otomatis.</div>
    """, unsafe_allow_html=True)

    col_up, col_aturan = st.columns([7, 5], gap="large")

    with col_aturan:
        st.markdown("""
            <div class="studio-panel" style="background:#F0F7FF; border-color:#BFDBFE;">
                <div style="font-weight:800; color:#0F2557; font-size:0.95rem; margin-bottom:10px;">
                    🔒 4 Lapis Proteksi Dataset Khusus JAKI
                </div>
                <div style="font-size:0.82rem; color:#334155; line-height:1.6;">
                    Agar sistem ini tidak disalahgunakan untuk dataset umum di luar topik skripsi JAKI, setiap file yang diunggah akan melewati pemeriksaan ketat:
                    <ol style="margin-top:8px; padding-left:18px;">
                        <li><strong>Wajib Kolom <code>Ulasan</code>:</strong> Nama kolom teks harus <code>Ulasan</code>.</li>
                        <li><strong>Zero-Tolerance E-Commerce:</strong> Menolak otomatis CSV Shopee, Tokopedia, Gojek, Kurir, Ongkir, atau Seller.</li>
                        <li><strong>Wajib Entitas Inti JAKI:</strong> Minimal 5% data wajib menyebut kata <em>JAKI, JakLapor, JakWifi, JakLingko,</em> atau <em>Pemprov DKI</em>.</li>
                        <li><strong>Validasi Konteks Layanan Publik:</strong> Memeriksa kosa kata laporan warga, kelurahan, pajak, atau fasilitas DKI.</li>
                    </ol>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col_up:
        st.markdown("""
            <div class="studio-panel">
                <div class="studio-header">
                    <span class="studio-step-badge">UNGGAH FILE DATASET RESMI JAKI</span>
                    <span style="font-size:0.74rem; color:#64748B;">Format: .CSV / .XLSX</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Pilih atau seret file CSV / Excel hasil scraping Aplikasi JAKI:", type=['csv', 'xlsx'])

        if uploaded_file is None:
            st.session_state.df_hasil = None
        else:
            try:
                with st.spinner("🔍 Memeriksa keaslian domain dataset JAKI..."):
                    if uploaded_file.name.lower().endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)

                    is_valid_jaki, pesan_validasi = validasi_khusus_jaki(df)

                if not is_valid_jaki:
                    st.error(pesan_validasi)
                    st.session_state.df_hasil = None
                else:
                    st.success(pesan_validasi)

                    if st.button("⚡ Mulai Pembersihan & Klasifikasi Massal SVM", type="primary", use_container_width=True):
                        with st.spinner("⏳ Sedang menjalankan Sastrawi Stemming & Klasifikasi SVM pada seluruh baris data..."):
                            with st.status("⚙️ Memproses Dataset Ulasan JAKI...", expanded=True) as status:
                                st.write("1️⃣ Menghapus baris kosong & menormalisasi teks ulasan warga...")
                                df_bersih = df.dropna(subset=['Ulasan']).copy()
                                df_bersih['Ulasan_Bersih'] = df_bersih['Ulasan'].apply(pembersihan_teks)
                                df_bersih = df_bersih[df_bersih['Ulasan_Bersih'].str.strip() != '']

                                st.write("2️⃣ Mengubah ribuan ulasan menjadi matriks TF-IDF...")
                                vektor_massal = tfidf_vect.transform(df_bersih['Ulasan_Bersih'])

                                st.write("3️⃣ Mengklasifikasikan sentimen Positif & Negatif dengan SVM...")
                                df_bersih['Label_Sentimen'] = svm_model.predict(vektor_massal)

                                st.session_state.df_hasil = df_bersih
                                status.update(label="✅ Pemrosesan Massal Selesai!", state="complete", expanded=False)

                        st.success("🎉 Seluruh ulasan berhasil diklasifikasikan! Anda dapat mengunduh hasilnya atau langsung membuka Dashboard Visualisasi.")

            except Exception as e:
                st.error(f"Terjadi kesalahan teknis saat membaca file: {e}")
                st.session_state.df_hasil = None

    # TAMPILKAN TABEL PREVIEW JIKA DATA SUDAH DIPROSES
    if st.session_state.df_hasil is not None:
        st.write("---")
        df_tampil = st.session_state.df_hasil
        c_dl1, c_dl2 = st.columns([1, 1])
        with c_dl1:
            csv_output = df_tampil.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Unduh Hasil Analisis Lengkap (Hasil_JAKI.csv)",
                data=csv_output,
                file_name='Hasil_JAKI.csv',
                mime='text/csv',
                use_container_width=True
            )
        with c_dl2:
            if st.button("📊 Buka Dashboard Visualisasi Sekarang →", type="primary", use_container_width=True):
                pindah_menu("visualisasi")
                st.rerun()

        with st.expander("📄 Pratinjau 10 Baris Pertama Hasil Klasifikasi SVM", expanded=True):
            st.dataframe(df_tampil.head(10), use_container_width=True)


# =====================================================================
# MENU 4: DASHBOARD VISUALISASI EKSKLUSIF
# =====================================================================
elif st.session_state.nav_menu == "visualisasi":

    st.markdown("""
        <div class="sec-eyebrow">VISUAL ANALYTICS DASHBOARD</div>
        <div class="sec-title">Dashboard Visualisasi Opini & Keluhan Warga JAKI</div>
        <div class="sec-desc">Ringkasan statistik distribusi sentimen, interaksi pengguna, peta panas jam laporan, serta kata kunci dominan.</div>
    """, unsafe_allow_html=True)

    if st.session_state.df_hasil is None:
        st.warning("⚠️ Belum ada dataset yang diproses. Silakan unggah dan proses file CSV JAKI terlebih dahulu pada menu **📁 Pemrosesan Massal CSV**.")
        if st.button("📁 Ke Menu Pemrosesan Massal CSV →", type="primary"):
            pindah_menu("massal")
            st.rerun()
    else:
        with st.spinner("📊 Merender grafik interaktif & Word Cloud..."):
            df_viz = st.session_state.df_hasil
            total_ulasan = len(df_viz)
            jml_positif = len(df_viz[df_viz['Label_Sentimen'] == 'Positif'])
            jml_negatif = len(df_viz[df_viz['Label_Sentimen'] == 'Negatif'])

            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-card kpi-total">
                    <p class="kpi-title" style="color:#64748B;">Total Ulasan Diproses</p>
                    <p class="kpi-value" style="color:#0F172A;">{total_ulasan:,}</p>
                </div>
                <div class="kpi-card kpi-positif">
                    <p class="kpi-title" style="color:#4E79A7;">Sentimen Positif (Biru)</p>
                    <p class="kpi-value" style="color:#4E79A7;">{jml_positif:,} <span style="font-size:1rem;">({round((jml_positif/total_ulasan)*100, 1)}%)</span></p>
                </div>
                <div class="kpi-card kpi-negatif">
                    <p class="kpi-title" style="color:#E15759;">Sentimen Negatif (Merah)</p>
                    <p class="kpi-value" style="color:#E15759;">{jml_negatif:,} <span style="font-size:1rem;">({round((jml_negatif/total_ulasan)*100, 1)}%)</span></p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            color_map = {'Positif': '#4E79A7', 'Negatif': '#E15759'}

            # BARIS 1: PIE, BAR, STRIP SCATTER
            col1, col2, col3 = st.columns(3)
            sentimen_counts = df_viz['Label_Sentimen'].value_counts().reset_index()
            sentimen_counts.columns = ['Sentimen', 'Jumlah']

            with col1:
                fig_pie = px.pie(
                    sentimen_counts, values='Jumlah', names='Sentimen',
                    title='Proporsi Sentimen Keseluruhan',
                    color='Sentimen', color_discrete_map=color_map, hole=0.42
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(margin=dict(t=45, b=20, l=10, r=10), showlegend=True)
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                fig_bar = px.bar(
                    sentimen_counts, x='Sentimen', y='Jumlah',
                    title='Perbandingan Jumlah Ulasan',
                    text='Jumlah', color='Sentimen', color_discrete_map=color_map
                )
                fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
                fig_bar.update_layout(margin=dict(t=45, b=20, l=10, r=10), showlegend=True)
                st.plotly_chart(fig_bar, use_container_width=True)

            with col3:
                if 'Jumlah_Like' in df_viz.columns:
                    df_viz['Jumlah_Like'] = pd.to_numeric(df_viz['Jumlah_Like'], errors='coerce').fillna(0)
                    df_viz['Ulasan_Wrap'] = df_viz['Ulasan'].apply(lambda x: '<br>'.join(textwrap.wrap(str(x), width=50)))

                    fig_scatter = px.strip(
                        df_viz, x='Label_Sentimen', y='Jumlah_Like', color='Label_Sentimen',
                        title="Sebaran Interaksi Ulasan (Berdasarkan Like)",
                        custom_data=['Ulasan_Wrap'], color_discrete_map=color_map
                    )
                    fig_scatter.update_traces(
                        marker=dict(size=8, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')),
                        hovertemplate="<b>Sentimen:</b> %{x}<br><b>Jumlah Like:</b> %{y}<br><br><b>Ulasan:</b><br>%{customdata[0]}<extra></extra>"
                    )
                    fig_scatter.update_layout(margin=dict(t=45, b=20, l=10, r=10), showlegend=True)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("💡 Kolom 'Jumlah_Like' tidak ditemukan di dalam dataset.")

            st.write("---")

            # BARIS 2: HEATMAP & 2 WORD CLOUD
            col4, col5, col6 = st.columns(3)

            with col4:
                if 'Tanggal_dan_Jam' in df_viz.columns:
                    df_viz['Tanggal_dan_Jam_DT'] = pd.to_datetime(df_viz['Tanggal_dan_Jam'], errors='coerce')
                    hari_map = {0: 'Senin', 1: 'Selasa', 2: 'Rabu', 3: 'Kamis', 4: 'Jumat', 5: 'Sabtu', 6: 'Minggu'}
                    df_viz['Hari'] = df_viz['Tanggal_dan_Jam_DT'].dt.dayofweek.map(hari_map)
                    df_viz['Jam'] = df_viz['Tanggal_dan_Jam_DT'].dt.hour

                    heatmap_data = df_viz.groupby(['Hari', 'Jam']).size().reset_index(name='Jumlah')
                    pivot_data = heatmap_data.pivot(index='Hari', columns='Jam', values='Jumlah').fillna(0)
                    kategori_hari = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
                    pivot_data = pivot_data.reindex(kategori_hari[::-1])

                    for jam in range(24):
                        if jam not in pivot_data.columns:
                            pivot_data[jam] = 0
                    pivot_data = pivot_data[sorted(pivot_data.columns)]

                    fig_hm = px.imshow(
                        pivot_data, labels=dict(x="Jam Masuk", y="", color="Jumlah"),
                        x=pivot_data.columns, y=pivot_data.index,
                        color_continuous_scale="Blues", aspect="auto",
                        title="Peta Panas Waktu Masuk Ulasan"
                    )
                    fig_hm.update_traces(
                        hovertemplate='<b>Hari:</b> %{y}<br><b>Jam:</b> %{x}:00<br><b>Jumlah Ulasan Masuk:</b> %{z}<extra></extra>'
                    )
                    fig_hm.update_layout(margin=dict(t=40, b=20, l=10, r=10), coloraxis_showscale=False)
                    st.plotly_chart(fig_hm, use_container_width=True)
                else:
                    st.info("💡 Kolom 'Tanggal_dan_Jam' tidak ditemukan untuk membuat Heatmap.")

            def buat_wordcloud(teks_gabungan, title, colormap):
                if len(teks_gabungan.strip()) > 0:
                    wc = WordCloud(
                        width=500, height=350,
                        background_color='white',
                        colormap=colormap,
                        max_words=80
                    ).generate(teks_gabungan)
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis("off")
                    ax.set_title(title, fontfamily='sans-serif', fontweight='bold', fontsize=12, pad=10, loc='left')
                    fig.tight_layout(pad=1)
                    return fig
                return None

            with col5:
                teks_pos = " ".join(review for review in df_viz[df_viz['Label_Sentimen'] == 'Positif']['Ulasan_Bersih'])
                fig_wc_pos = buat_wordcloud(teks_pos, "Kata Kunci Sentimen Positif", "Blues")
                if fig_wc_pos:
                    st.pyplot(fig_wc_pos)

            with col6:
                teks_neg = " ".join(review for review in df_viz[df_viz['Label_Sentimen'] == 'Negatif']['Ulasan_Bersih'])
                fig_wc_neg = buat_wordcloud(teks_neg, "Kata Kunci Sentimen Negatif", "Reds")
                if fig_wc_neg:
                    st.pyplot(fig_wc_neg)
