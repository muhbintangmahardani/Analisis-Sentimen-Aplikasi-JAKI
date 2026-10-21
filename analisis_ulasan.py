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
    page_title="JAKI Sentiment Analytics | Pemprov DKI",
    page_icon="https://jaki.jakarta.go.id/favicon.ico",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================================
# 2. CSS KUSTOM: GAYA CLEAN MUSAVISION + TEMA BIRU JAKI (NO AI SLOP)
# =====================================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #F8FAFC;
        color: #0F172A;
    }

    /* Sembunyikan dekorasi atas bawaan Streamlit agar tidak terpotong */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 1.5rem !important;
    }
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 4rem !important;
        max-width: 1240px !important;
    }

    /* TOP STATUS BAR (Dark Navy Bar ala MusaVision) */
    .mv-topbar {
        background-color: #0B1736;
        color: #94A3B8;
        padding: 8px 20px;
        border-radius: 8px;
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        font-size: 0.74rem;
        margin-bottom: 12px;
        border: 1px solid #1E293B;
    }
    .mv-status-dot {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: rgba(37, 99, 235, 0.2);
        color: #60A5FA;
        border: 1px solid rgba(96, 165, 250, 0.3);
        padding: 2px 9px;
        border-radius: 99px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 600;
        margin-right: 10px;
    }

    /* NAVBAR BRAND HEADER */
    .mv-navbar {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 14px 24px;
        margin-bottom: 14px;
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
    }
    .mv-brand {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .mv-brand img {
        height: 34px;
        width: auto;
        object-fit: contain;
    }
    .mv-version {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 600;
        color: #0055B8;
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        padding: 2px 7px;
        border-radius: 4px;
    }

    /* HERO SECTION (MusaVision Clean Editorial Style) */
    .mv-hero-left {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 30px;
        height: 100%;
    }
    .mv-pill {
        display: inline-block;
        background-color: #EFF6FF;
        color: #0055B8;
        border: 1px solid #BFDBFE;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 4px 11px;
        border-radius: 99px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 16px;
    }
    .mv-hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.18;
        letter-spacing: -0.02em;
        margin-bottom: 14px;
    }
    .mv-hero-title span {
        color: #0055B8;
        font-weight: 700;
        font-style: italic;
    }
    .mv-hero-desc {
        font-size: 0.92rem;
        color: #475569;
        line-height: 1.65;
        margin-bottom: 22px;
    }
    .mv-code-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        background-color: #F1F5F9;
        color: #0F172A;
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid #E2E8F0;
    }

    /* 4 METRIC BOXES DI BAWAH HERO */
    .mv-stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin-top: 20px;
        padding-top: 20px;
        border-top: 1px solid #F1F5F9;
    }
    .mv-stat-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 14px;
    }
    .mv-stat-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
    }
    .mv-stat-val.accent {
        color: #0055B8;
    }
    .mv-stat-lbl {
        font-size: 0.68rem;
        color: #64748B;
        margin-top: 3px;
        font-weight: 500;
    }

    /* KARTU PREVIEW KANAN HERO */
    .mv-preview-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 22px;
        height: 100%;
    }
    .mv-preview-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 12px;
        margin-bottom: 14px;
        border-bottom: 1px solid #F1F5F9;
        font-size: 0.75rem;
        font-weight: 700;
        color: #334155;
    }
    .mv-preview-screen {
        background-color: #0B1736;
        border-radius: 10px;
        padding: 18px;
        color: #F8FAFC;
        margin-bottom: 14px;
    }
    .mv-preview-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.76rem;
        color: #64748B;
        padding-top: 8px;
    }

    /* SECTION HEADER ALA MUSAVISION */
    .mv-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #0055B8;
        margin-bottom: 4px;
    }
    .mv-sec-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.01em;
        margin-bottom: 4px;
    }
    .mv-sec-sub {
        font-size: 0.86rem;
        color: #64748B;
        margin-bottom: 20px;
    }

    /* 5 PIPELINE CARDS & TAXONOMY CARDS */
    .mv-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px 18px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .mv-card-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.66rem;
        font-weight: 700;
        color: #0055B8;
        background-color: #EFF6FF;
        border: 1px solid #DBEAFE;
        padding: 2px 8px;
        border-radius: 4px;
        display: inline-block;
        margin-bottom: 12px;
        width: fit-content;
    }
    .mv-card-title {
        font-size: 0.96rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 6px;
    }
    .mv-card-body {
        font-size: 0.8rem;
        color: #475569;
        line-height: 1.55;
        margin-bottom: 14px;
    }
    .mv-card-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: #64748B;
        padding-top: 10px;
        border-top: 1px solid #F1F5F9;
        display: flex;
        justify-content: space-between;
    }

    /* STUDIO PANEL (2 Kolom Input & Output) */
    .mv-studio-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px 22px;
        margin-bottom: 14px;
    }
    .mv-studio-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 12px;
        margin-bottom: 14px;
        border-bottom: 1px solid #F1F5F9;
    }
    .mv-num-circle {
        background-color: #0B1736;
        color: #FFFFFF;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 6px;
    }
    .mv-empty-state {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 56px 28px;
        text-align: center;
        color: #64748B;
    }

    /* HASIL SENTIMEN BOX */
    .mv-result-pos {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 5px solid #16A34A;
        border-radius: 10px;
        padding: 18px 20px;
        color: #14532D;
        margin-bottom: 14px;
    }
    .mv-result-neg {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 5px solid #DC2626;
        border-radius: 10px;
        padding: 18px 20px;
        color: #7F1D1D;
        margin-bottom: 14px;
    }

    /* BUTTON STYLING */
    div.stButton > button[kind="primary"] {
        background-color: #0B1736 !important;
        color: #FFFFFF !important;
        border: 1px solid #0B1736 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
        padding: 0.55rem 1.1rem !important;
        transition: all 0.15s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #0055B8 !important;
        border-color: #0055B8 !important;
    }
    div.stButton > button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #334155 !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
        padding: 0.55rem 1rem !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        border-color: #0055B8 !important;
        color: #0055B8 !important;
        background-color: #F8FAFC !important;
    }

    /* KPI CARDS DASHBOARD */
    .kpi-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-card {
        flex: 1;
        min-width: 220px;
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        text-align: left;
    }
    .kpi-total { border-left: 5px solid #64748B; }
    .kpi-positif { border-left: 5px solid #4E79A7; }
    .kpi-negatif { border-left: 5px solid #E15759; }
    .kpi-title {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin: 0 0 6px 0;
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# 3. INISIALISASI SASTRAWI & NLTK
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
    jml_fitur_tfidf = len(tfidf_vect.vocabulary_) if hasattr(tfidf_vect, 'vocabulary_') else 2500
except Exception:
    st.error("Gagal memuat file .pkl. Pastikan model_svm_jaki.pkl dan tfidf_jaki.pkl berada di direktori yang sama.")
    st.stop()

# =====================================================================
# 4. STATE NAVIGASI
# =====================================================================
if 'df_hasil' not in st.session_state:
    st.session_state.df_hasil = None
if 'nav_aktif' not in st.session_state:
    st.session_state.nav_aktif = "ringkasan"
if 'teks_uji' not in st.session_state:
    st.session_state.teks_uji = ""

def set_nav(menu):
    st.session_state.nav_aktif = menu

# =====================================================================
# 5. TOP STATUS BAR & BRAND NAVBAR (DENGAN LOGO JAKI CDN)
# =====================================================================
LOGO_JAKI_CDN = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/eb/Coat_of_arms_of_Jakarta.svg/240px-Coat_of_arms_of_Jakarta.svg.png"

st.markdown(f"""
    <div class="mv-topbar">
        <div>
            <span class="mv-status-dot">● SVM ENGINE READY</span>
            <span>Arsitektur Klasifikasi Sentimen Ulasan Aplikasi JAKI (Sastrawi + TF-IDF + Linear SVM)</span>
        </div>
        <div style="font-family:'JetBrains Mono', monospace; font-size:0.7rem;">
            Vocabulary: <strong style="color:#FFF;">{jml_fitur_tfidf:,} Token</strong> &nbsp;|&nbsp;
            Metode: <strong style="color:#60A5FA;">Support Vector Machine</strong> &nbsp;|&nbsp;
            Peneliti: <strong style="color:#FFF;">Artha Nugraha</strong>
        </div>
    </div>
    <div class="mv-navbar">
        <div class="mv-brand">
            <img src="{LOGO_JAKI_CDN}" alt="Logo DKI Jakarta">
            <div>
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="font-weight:800; font-size:1.05rem; color:#0F172A; letter-spacing:-0.01em;">JAKISentiment</span>
                    <span class="mv-version">v1.0 SVM</span>
                </div>
                <div style="font-size:0.74rem; color:#64748B;">Sistem Evaluasi Opini Publik Aplikasi Jakarta Kini (JAKI)</div>
            </div>
        </div>
        <div style="font-size:0.76rem; color:#64748B; font-weight:500;">
            Gunakan navigasi kolom di bawah untuk berpindah modul:
        </div>
    </div>
""", unsafe_allow_html=True)

# NAVBAR 4 KOLOM
c_nav1, c_nav2, c_nav3, c_nav4 = st.columns(4)
with c_nav1:
    if st.button(
        "Ringkasan & Arsitektur",
        type="primary" if st.session_state.nav_aktif == "ringkasan" else "secondary",
        use_container_width=True
    ):
        set_nav("ringkasan")
        st.rerun()
with c_nav2:
    if st.button(
        "Studio Analisis Teks Tunggal",
        type="primary" if st.session_state.nav_aktif == "tunggal" else "secondary",
        use_container_width=True
    ):
        set_nav("tunggal")
        st.rerun()
with c_nav3:
    if st.button(
        "Pemrosesan Massal (CSV)",
        type="primary" if st.session_state.nav_aktif == "massal" else "secondary",
        use_container_width=True
    ):
        set_nav("massal")
        st.rerun()
with c_nav4:
    label_viz = "Dashboard Visualisasi (Aktif)" if st.session_state.df_hasil is not None else "Dashboard Visualisasi"
    if st.button(
        label_viz,
        type="primary" if st.session_state.nav_aktif == "visualisasi" else "secondary",
        use_container_width=True
    ):
        set_nav("visualisasi")
        st.rerun()

st.write("")

# =====================================================================
# MODUL 1: RINGKASAN & ARSITEKTUR (LANDING PAGE GAYA MUSAVISION)
# =====================================================================
if st.session_state.nav_aktif == "ringkasan":

    hero_l, hero_r = st.columns([7, 5], gap="medium")

    with hero_l:
        st.markdown(f"""
            <div class="mv-hero-left">
                <div class="mv-pill">● Natural Language Processing &bull; Jakarta Smart City</div>
                <div class="mv-hero-title">
                    Klasifikasi Sentimen Ulasan Layanan Publik <span>(Aplikasi JAKI)</span>
                </div>
                <div class="mv-hero-desc">
                    Sistem analisis opini dan keluhan warga DKI Jakarta pada aplikasi <strong>JAKI</strong> menggunakan
                    normalisasi bahasa tidak baku, <em>stemming</em> Bahasa Indonesia <span class="mv-code-tag">Sastrawi</span>,
                    ekstraksi bobot kata <span class="mv-code-tag">TF-IDF</span>, serta pemisahan <em>hyperplane</em>
                    <span class="mv-code-tag">Support Vector Machine</span>.
                </div>
                <div class="mv-stat-grid">
                    <div class="mv-stat-box">
                        <div class="mv-stat-val">{jml_fitur_tfidf:,}</div>
                        <div class="mv-stat-lbl">Kosakata TF-IDF</div>
                    </div>
                    <div class="mv-stat-box">
                        <div class="mv-stat-val accent">2 Kelas</div>
                        <div class="mv-stat-lbl">Positif & Negatif</div>
                    </div>
                    <div class="mv-stat-box">
                        <div class="mv-stat-val">Linear</div>
                        <div class="mv-stat-lbl">Kernel Margin SVM</div>
                    </div>
                    <div class="mv-stat-box">
                        <div class="mv-stat-val accent">~10 ms</div>
                        <div class="mv-stat-lbl">Inferensi Teks</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        b_cta1, b_cta2 = st.columns(2)
        with b_cta1:
            if st.button("Buka Studio Analisis Teks →", type="primary", use_container_width=True):
                set_nav("tunggal")
                st.rerun()
        with b_cta2:
            if st.button("Unggah Dataset CSV JAKI →", type="secondary", use_container_width=True):
                set_nav("massal")
                st.rerun()

    with hero_r:
        st.markdown(f"""
            <div class="mv-preview-card">
                <div class="mv-preview-top">
                    <span style="color:#0055B8;">● Simulasi Inferensi Model SVM</span>
                    <span style="font-family:'JetBrains Mono', monospace; font-size:0.68rem; color:#64748B;">model_svm_jaki.pkl</span>
                </div>
                <div class="mv-preview-screen">
                    <div style="font-family:'JetBrains Mono', monospace; font-size:0.68rem; color:#93C5FD; margin-bottom:6px;">
                        INPUT ULASAN WARGA:
                    </div>
                    <div style="font-size:0.84rem; color:#F8FAFC; line-height:1.5; background:rgba(255,255,255,0.06); padding:10px 12px; border-radius:6px; margin-bottom:12px; border:1px solid rgba(255,255,255,0.1);">
                        "Aplikasi JAKI lemot banget pas mau lapor jalan rusak di kelurahan, sering error!"
                    </div>
                    <div style="font-family:'JetBrains Mono', monospace; font-size:0.68rem; color:#93C5FD; margin-bottom:6px;">
                        TOKEN SASTRAWI & NORMALISASI:
                    </div>
                    <div style="font-family:'JetBrains Mono', monospace; font-size:0.75rem; color:#BFDBFE; background:rgba(0,85,184,0.28); padding:9px 12px; border-radius:6px; border:1px dashed #60A5FA; margin-bottom:14px;">
                        aplikasi jaki <strong>lambat</strong> lapor jalan rusak lurah sering <strong>rusak</strong>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center; background:rgba(220,38,38,0.2); border:1px solid #F87171; padding:10px 14px; border-radius:8px;">
                        <span style="font-size:0.76rem; font-weight:700; color:#FCA5A5;">PREDIKSI: SENTIMEN NEGATIF</span>
                        <span style="font-family:'JetBrains Mono', monospace; font-size:0.78rem; font-weight:700; color:#FFF;">Conf: 94.6%</span>
                    </div>
                </div>
                <div class="mv-preview-footer">
                    <span>Kategori: <strong>Keluhan Teknis & Layanan</strong></span>
                    <span style="color:#0055B8; font-weight:700;">✓ Tervalidasi</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("---")

    # LIMA TAHAP PIPELINE KOMPUTASI (Panduan Orang Awam)
    st.markdown("""
        <div class="mv-eyebrow">ARSITEKTUR & METODOLOGI</div>
        <div class="mv-sec-title">Lima Tahap Pipeline Komputasi Analisis Sentimen</div>
        <div class="mv-sec-sub">Penjelasan berurutan bagi pengguna awam mengenai cara kerja sistem mengubah teks ulasan warga menjadi keputusan sentimen:</div>
    """, unsafe_allow_html=True)

    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        st.markdown("""
            <div class="mv-card">
                <div>
                    <div class="mv-card-tag">01 / INPUT GUARD</div>
                    <div class="mv-card-title">Validasi Konteks JAKI</div>
                    <div class="mv-card-body">Memeriksa kesesuaian topik ulasan agar hanya memproses data layanan aplikasi JAKI dan menolak dataset e-commerce.</div>
                </div>
                <div class="mv-card-meta"><span>Filter Domain</span><span>Heuristic</span></div>
            </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown("""
            <div class="mv-card">
                <div>
                    <div class="mv-card-tag">02 / NORMALISASI</div>
                    <div class="mv-card-title">Case Folding & Slang</div>
                    <div class="mv-card-body">Menyeragamkan huruf kecil, menghapus simbol/URL, dan membakukan kata gaul (<em>"lemot/ngelag"</em> menjadi <em>"lambat"</em>).</div>
                </div>
                <div class="mv-card-meta"><span>Kamus Baku</span><span>Regex</span></div>
            </div>
        """, unsafe_allow_html=True)
    with p3:
        st.markdown("""
            <div class="mv-card">
                <div>
                    <div class="mv-card-tag">03 / PREPROCESSING</div>
                    <div class="mv-card-title">Stopwords & Sastrawi</div>
                    <div class="mv-card-body">Menghapus kata hubung umum tanpa membuang kata negasi (<em>"tidak, belum, jangan"</em>) lalu mengubahnya ke kata dasar.</div>
                </div>
                <div class="mv-card-meta"><span>Stemmer</span><span>NLTK + Sastrawi</span></div>
            </div>
        """, unsafe_allow_html=True)
    with p4:
        st.markdown("""
            <div class="mv-card">
                <div>
                    <div class="mv-card-tag">04 / EKSTRAKSI FITUR</div>
                    <div class="mv-card-title">Vektorisasi TF-IDF</div>
                    <div class="mv-card-body">Mengubah kumpulan kata dasar menjadi matriks bobot numerik berdasarkan frekuensi kemunculan kata di dalam dokumen.</div>
                </div>
                <div class="mv-card-meta"><span>Matriks</span><span>tfidf_jaki.pkl</span></div>
            </div>
        """, unsafe_allow_html=True)
    with p5:
        st.markdown("""
            <div class="mv-card">
                <div>
                    <div class="mv-card-tag">05 / KLASIFIKASI</div>
                    <div class="mv-card-title">Hyperplane SVM</div>
                    <div class="mv-card-body">Menghitung jarak margin terhadap bidang pemisah terbaik untuk menentukan kelas sentimen Positif atau Negatif.</div>
                </div>
                <div class="mv-card-meta"><span>Classifier</span><span>Sigmoid Margin</span></div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("---")

    # TAKSONOMI TARGET & PANDUAN PENGGUNAAN (Gaya MusaVision)
    st.markdown("""
        <div class="mv-eyebrow">TAKSONOMI KELAS SENTIMEN & PANDUAN</div>
        <div class="mv-sec-title">Karakteristik Kelas Sentimen & Cara Penggunaan</div>
        <div class="mv-sec-sub">Pola bahasa yang dikenali oleh model pada ulasan pengguna aplikasi JAKI:</div>
    """, unsafe_allow_html=True)

    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("""
            <div class="mv-card">
                <div>
                    <div class="mv-card-tag" style="background:#ECFDF5; color:#059669; border-color:#A7F3D0;">KELAS TARGET 01</div>
                    <div class="mv-card-title">Sentimen Positif (Apresiasi Warga)</div>
                    <div class="mv-card-body">Mencakup ulasan kepuasan terhadap kecepatan tindak lanjut petugas JakLapor, kemudahan cek pajak/KTP, dan fitur layanan publik DKI yang membantu warga.</div>
                </div>
                <div class="mv-card-meta"><span>Indikator Warna</span><span style="color:#0055B8; font-weight:700;">Biru (#4E79A7)</span></div>
            </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown("""
            <div class="mv-card">
                <div>
                    <div class="mv-card-tag" style="background:#FEF2F2; color:#DC2626; border-color:#FECACA;">KELAS TARGET 02</div>
                    <div class="mv-card-title">Sentimen Negatif (Keluhan & Kendala)</div>
                    <div class="mv-card-body">Mencakup keluhan teknis aplikasi (lambat, gagal masuk, layar kosong/error) maupun keluhan atas lambatnya respon penanganan laporan di lapangan.</div>
                </div>
                <div class="mv-card-meta"><span>Indikator Warna</span><span style="color:#DC2626; font-weight:700;">Merah (#E15759)</span></div>
            </div>
        """, unsafe_allow_html=True)
    with t3:
        st.markdown("""
            <div class="mv-card" style="background-color:#0B1736; color:#FFFFFF; border-color:#0B1736;">
                <div>
                    <div class="mv-card-tag" style="background:rgba(255,255,255,0.12); color:#93C5FD; border-color:transparent;">PANDUAN CEPAT AWAM</div>
                    <div class="mv-card-title" style="color:#FFFFFF;">Mulai Uji Model Sekarang</div>
                    <div class="mv-card-body" style="color:#CBD5E1;">Pilih menu <strong>Studio Analisis Teks Tunggal</strong> di atas untuk menguji satu kalimat, atau buka <strong>Pemrosesan Massal (CSV)</strong> untuk menganalisis tabel data scraping.</div>
                </div>
                <div class="mv-card-meta" style="border-top-color:rgba(255,255,255,0.12); color:#93C5FD;"><span>Status Sistem</span><span>Siap Digunakan</span></div>
            </div>
        """, unsafe_allow_html=True)


# =====================================================================
# MODUL 2: STUDIO ANALISIS TEKS TUNGGAL (2 KOLOM ALA MUSAVISION)
# =====================================================================
elif st.session_state.nav_aktif == "tunggal":

    st.markdown("""
        <div class="mv-eyebrow">STUDIO PENGUJIAN MODEL</div>
        <div class="mv-sec-title">Analisis Sentimen Teks Tunggal & Diagnosis NLP</div>
        <div class="mv-sec-sub">Pilih salah satu sampel ulasan di bawah atau ketik kalimat ulasan secara manual untuk melihat hasil klasifikasi SVM beserta tahapan pembersihan katanya.</div>
    """, unsafe_allow_html=True)

    # BARIS SAMPEL CEPAT UNTUK ORANG AWAM
    st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#334155; margin-bottom:8px;'>1. Galeri Sampel Ulasan Warga (Klik untuk mengisi otomatis):</div>", unsafe_allow_html=True)
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    with sc1:
        if st.button("Apresiasi JakLapor", use_container_width=True):
            st.session_state.teks_uji = "Aplikasi JAKI sangat membantu warga DKI, kemarin saya lapor jalan berlubang di kelurahan langsung ditindak cepat oleh petugas!"
    with sc2:
        if st.button("Keluhan Aplikasi Lemot", use_container_width=True):
            st.session_state.teks_uji = "Aplikasi JAKI lemot banget pas mau daftar vaksin dan sering error blank putih, tolong segera diperbaiki servernya!"
    with sc3:
        if st.button("Kemudahan Layanan DKI", use_container_width=True):
            st.session_state.teks_uji = "Keren banget aplikasi JAKI sekarang, cek informasi pajak dan fasilitas umum di Jakarta jadi jauh lebih praktis dan bagus."
    with sc4:
        if st.button("Keluhan Respon Lambat", use_container_width=True):
            st.session_state.teks_uji = "Percuma lapor parkir liar dan sampah lewat JAKI kalau respon petugasnya lelet banget, aplikasinya juga sering ngelag."
    with sc5:
        if st.button("Reset Masukan", use_container_width=True):
            st.session_state.teks_uji = ""

    st.write("")

    # 2 KOLOM STUDIO (Kiri: Input Teks, Kanan: Hasil Diagnosis)
    st_kiri, st_kanan = st.columns([5, 7], gap="medium")

    with st_kiri:
        st.markdown("""
            <div class="mv-studio-box">
                <div class="mv-studio-head">
                    <span class="mv-num-circle">2. MASUKAN TEKS ULASAN</span>
                    <span style="font-family:'JetBrains Mono', monospace; font-size:0.7rem; color:#64748B;">ID / Slang Jakarta</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        user_input = st.text_area(
            "Ketik keluhan atau ulasan pengguna JAKI secara manual:",
            value=st.session_state.teks_uji,
            placeholder="Contoh: Aplikasi JAKI lemot banget pas mau daftar vaksin, tolong diperbaiki!",
            height=150
        )

        btn_prediksi = st.button("Jalankan Prediksi Sentimen", type="primary", use_container_width=True)

    with st_kanan:
        st.markdown("""
            <div class="mv-studio-box">
                <div class="mv-studio-head">
                    <span class="mv-num-circle">3. HASIL KLASIFIKASI & BEDAH PREPROCESSING</span>
                    <span style="font-family:'JetBrains Mono', monospace; font-size:0.7rem; color:#0055B8;">SVM Decision Margin</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if not btn_prediksi:
            st.markdown("""
                <div class="mv-empty-state">
                    <div style="font-weight:700; color:#0F172A; font-size:0.96rem; margin-bottom:6px;">Menunggu Masukan Teks Ulasan</div>
                    <div style="font-size:0.82rem; max-width:420px; margin:0 auto; line-height:1.55;">
                        Pilih salah satu sampel cepat di atas atau ketik kalimat ulasan di sebelah kiri, kemudian tekan tombol
                        <strong>Jalankan Prediksi Sentimen</strong> untuk menampilkan hasil analisis.
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            if user_input.strip() == "":
                st.warning("Mohon masukkan teks ulasan terlebih dahulu pada kolom sebelah kiri.")
            else:
                with st.spinner("Menjalankan normalisasi kata, stemming Sastrawi, dan klasifikasi SVM..."):
                    time.sleep(0.35)
                    teks_bersih = pembersihan_teks(user_input)
                    teks_vektor = tfidf_vect.transform([teks_bersih])
                    prediction = svm_model.predict(teks_vektor)[0]
                    jarak_margin = float(np.ravel(svm_model.decision_function(teks_vektor))[0])
                    probabilitas = 1 / (1 + np.exp(-jarak_margin))
                    keyakinan = probabilitas if probabilitas > 0.5 else (1 - probabilitas)
                    persentase_yakin = round(keyakinan * 100, 1)

                if prediction == 'Positif':
                    st.markdown(f"""
                        <div class="mv-result-pos">
                            <div style="font-family:'JetBrains Mono', monospace; font-size:0.7rem; font-weight:700; text-transform:uppercase;">Hasil Prediksi Model SVM</div>
                            <div style="font-size:1.45rem; font-weight:800; margin:4px 0;">Sentimen POSITIF (Keyakinan: {persentase_yakin}%)</div>
                            <div style="font-size:0.84rem;">Ulasan terklasifikasi sebagai bentuk apresiasi atau kepuasan terhadap layanan aplikasi JAKI.</div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="mv-result-neg">
                            <div style="font-family:'JetBrains Mono', monospace; font-size:0.7rem; font-weight:700; text-transform:uppercase;">Hasil Prediksi Model SVM</div>
                            <div style="font-size:1.45rem; font-weight:800; margin:4px 0;">Sentimen NEGATIF (Keyakinan: {persentase_yakin}%)</div>
                            <div style="font-size:0.84rem;">Ulasan terklasifikasi sebagai keluhan teknis aplikasi atau ketidakpuasan layanan publik.</div>
                        </div>
                    """, unsafe_allow_html=True)

                st.progress(int(min(max(persentase_yakin, 1), 100)), text=f"Confidence Score: {persentase_yakin}% | Hyperplane Margin: {jarak_margin:.3f}")

                st.markdown(f"""
                    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:14px 16px; margin-top:10px; font-size:0.83rem;">
                        <div style="font-weight:700; color:#0F172A; margin-bottom:4px;">Teks Setelah Dibersihkan (Sastrawi Stemmer):</div>
                        <div style="font-family:'JetBrains Mono', monospace; color:#0055B8; background:#F8FAFC; padding:8px 10px; border-radius:6px; border:1px solid #E2E8F0;">
                            {teks_bersih if teks_bersih else "(Teks habis terfilter stopwords)"}
                        </div>
                    </div>
                """, unsafe_allow_html=True)


# =====================================================================
# MODUL 3: PEMROSESAN MASSAL (MENGGUNAKAN FILTER ASLI ANALISIS_ULASAN_2.PY)
# =====================================================================
elif st.session_state.nav_aktif == "massal":

    st.markdown("""
        <div class="mv-eyebrow">BATCH DATASET PROCESSING</div>
        <div class="mv-sec-title">Pemrosesan Massal Dataset Ulasan JAKI</div>
        <div class="mv-sec-sub">Unggah file CSV atau Excel hasil scraping ulasan aplikasi JAKI untuk dianalisis secara massal dan divisualisasikan ke dalam dashboard.</div>
    """, unsafe_allow_html=True)

    col_up_left, col_up_right = st.columns([7, 5], gap="medium")

    with col_up_right:
        st.markdown("""
            <div class="mv-card">
                <div>
                    <div class="mv-card-tag">SPESIFIKASI FILTER DATASET</div>
                    <div class="mv-card-title">Ketentuan Validasi Konteks JAKI</div>
                    <div class="mv-card-body">
                        Sistem ini menerapkan <strong>Validasi Konteks Berlapis (Keyword Heuristics & TF-IDF Overlap)</strong>:
                        <ul style="margin-top:8px; padding-left:18px; line-height:1.6;">
                            <li>Mendukung kolom bernama <code>Ulasan</code>, <code>Review</code>, <code>Komentar</code>, atau <code>Text</code>.</li>
                            <li>Menghitung perbandingan skor kata kunci utama JAKI (<em>jaki, jaklapor, jakwifi, pemprov dki, kelurahan</em>) terhadap kata kunci domain lain.</li>
                            <li>Dataset resmi JAKI tetap diterima meskipun ada sedikit kata umum, namun dataset murni e-commerce/ojol (Shopee, Tokopedia, Gojek) akan ditolak.</li>
                        </ul>
                    </div>
                </div>
                <div class="mv-card-meta"><span>Metode Validasi</span><span>Heuristic Scoring + TF-IDF</span></div>
            </div>
        """, unsafe_allow_html=True)

    with col_up_left:
        st.markdown("""
            <div class="mv-studio-box">
                <div class="mv-studio-head">
                    <span class="mv-num-circle">UNGGAH FILE DATASET (.CSV / .XLSX)</span>
                    <span style="font-family:'JetBrains Mono', monospace; font-size:0.7rem; color:#64748B;">Drag & Drop</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Pilih file CSV atau Excel hasil scraping:", type=['csv', 'xlsx'])

        if uploaded_file is None:
            st.session_state.df_hasil = None
        else:
            try:
                with st.spinner("Memeriksa struktur kolom dan memvalidasi konteks dataset JAKI..."):
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)

                    # 1. FLEKSIBILITAS NAMA KOLOM (DARI KODE ASLI ANALISIS_ULASAN_2.PY ANDA)
                    kandidat_kolom = ['ulasan', 'review', 'reviews', 'komentar', 'comment', 'text', 'content']
                    kolom_terpilih = None

                    for col in df.columns:
                        if str(col).strip().lower() in kandidat_kolom:
                            kolom_terpilih = col
                            break

                if kolom_terpilih is None:
                    st.error("❌ Akses Ditolak: Kolom ulasan tidak ditemukan. Harap gunakan kolom bernama 'Ulasan', 'Review', 'Komentar', atau 'Text'.")
                    st.session_state.df_hasil = None
                else:
                    # Samakan nama kolom menjadi 'Ulasan' untuk pemrosesan internal
                    df = df.rename(columns={kolom_terpilih: 'Ulasan'})

                    # 2. VALIDASI KONTEKS BERLAPIS (PERSIS DARI ANALISIS_ULASAN_2.PY ANDA)
                    teks_gabungan = " ".join(df['Ulasan'].dropna().astype(str).tolist()).lower()

                    # A. Pembobotan Kata Kunci JAKI
                    kw_utama = ['jaki', 'jakarta kini', 'pemprov dki', 'dki jakarta', 'beritajakarta']
                    kw_fitur = ['jaklapor', 'jakpantau', 'jakwifi', 'jakwarta', 'jakrespon', 'kelurahan', 'kecamatan', 'rt/rw', 'lapor', 'crm', 'posyandu', 'pbb']

                    skor_utama = sum(teks_gabungan.count(w) for w in kw_utama) * 3
                    skor_fitur = sum(teks_gabungan.count(w) for w in kw_fitur) * 1
                    total_skor_jaki = skor_utama + skor_fitur

                    # B. Anti-Keywords (Domain Lain)
                    kw_domain_lain = ['driver', 'gojek', 'gofood', 'shopee', 'tokopedia', 'grab', 'pln', 'ongkir', 'pesanan', 'm-banking', 'seller']
                    skor_domain_lain = sum(teks_gabungan.count(w) for w in kw_domain_lain)

                    # C. Check Kesesuaian Kosa Kata TF-IDF
                    sampel_teks = df['Ulasan'].dropna().astype(str).head(100)
                    sampel_vektor = tfidf_vect.transform(sampel_teks)
                    jumlah_fitur_match = (sampel_vektor.sum(axis=0) > 0).sum()

                    # EVALUASI GERBANG KEAMANAN
                    if skor_domain_lain > total_skor_jaki and skor_domain_lain > 5:
                        st.error("❌ Akses Ditolak (Keamanan Konteks): Terdeteksi ulasan dari aplikasi e-commerce/ride-hailing, bukan konteks aplikasi JAKI.")
                        st.session_state.df_hasil = None
                    elif total_skor_jaki < 3 and jumlah_fitur_match < 5:
                        st.error("❌ Akses Ditolak (Keamanan Konteks): File yang diunggah tidak mengandung kosa kata/entitas terkait pelayanan aplikasi JAKI.")
                        st.session_state.df_hasil = None
                    else:
                        st.success(f"✅ File tervalidasi sebagai dataset JAKI! Siap memproses data dari kolom '{kolom_terpilih}' ({df.shape[0]:,} baris).")

                        if st.button("Mulai Pembersihan & Analisis Massal", type="primary", use_container_width=True):
                            with st.spinner("Menjalankan Sastrawi Stemming & Klasifikasi SVM pada seluruh baris data..."):
                                with st.status("Memproses Data Ulasan JAKI...", expanded=True) as status:
                                    st.write("Tahap 1: Membersihkan teks dan menjalankan stemming Sastrawi...")
                                    df_bersih = df.dropna(subset=['Ulasan']).copy()
                                    df_bersih['Ulasan_Bersih'] = df_bersih['Ulasan'].apply(pembersihan_teks)
                                    df_bersih = df_bersih[df_bersih['Ulasan_Bersih'].str.strip() != '']

                                    st.write("Tahap 2: Transformasi matriks TF-IDF dan prediksi kelas SVM...")
                                    vektor_massal = tfidf_vect.transform(df_bersih['Ulasan_Bersih'])
                                    df_bersih['Label_Sentimen'] = svm_model.predict(vektor_massal)

                                    st.session_state.df_hasil = df_bersih
                                    status.update(label="Pemrosesan Selesai!", state="complete", expanded=False)

                            st.success("Berhasil! Silakan unduh hasil analisis di bawah atau buka modul **Dashboard Visualisasi**.")

            except Exception as e:
                st.error(f"Terjadi kesalahan teknis: {e}")
                st.session_state.df_hasil = None

    if st.session_state.df_hasil is not None:
        st.write("---")
        df_out = st.session_state.df_hasil
        act1, act2 = st.columns(2)
        with act1:
            csv_output = df_out.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Unduh Hasil Analisis (Hasil_JAKI.csv)",
                data=csv_output,
                file_name='Hasil_JAKI.csv',
                mime='text/csv',
                use_container_width=True
            )
        with act2:
            if st.button("Buka Dashboard Visualisasi →", type="primary", use_container_width=True):
                set_nav("visualisasi")
                st.rerun()

        with st.expander("Klik untuk melihat 10 sampel data pertama yang telah diproses", expanded=True):
            st.dataframe(df_out.head(10), use_container_width=True)


# =====================================================================
# MODUL 4: DASHBOARD VISUALISASI EKSKLUSIF
# =====================================================================
elif st.session_state.nav_aktif == "visualisasi":

    st.markdown("""
        <div class="mv-eyebrow">VISUAL ANALYTICS DASHBOARD</div>
        <div class="mv-sec-title">Dashboard Visualisasi Sentimen Aplikasi JAKI</div>
        <div class="mv-sec-sub">Distribusi proporsi sentimen, sebaran interaksi ulasan, peta panas jam masuk laporan, serta kata kunci dominan.</div>
    """, unsafe_allow_html=True)

    if st.session_state.df_hasil is None:
        st.warning("Silakan unggah dan proses data di modul **Pemrosesan Massal (CSV)** terlebih dahulu untuk mengaktifkan Dashboard.")
        if st.button("Ke Pemrosesan Massal (CSV) →", type="primary"):
            set_nav("massal")
            st.rerun()
    else:
        with st.spinner("Memuat visualisasi grafik dan Word Cloud..."):
            df_viz = st.session_state.df_hasil
            total_ulasan = len(df_viz)
            jml_positif = len(df_viz[df_viz['Label_Sentimen'] == 'Positif'])
            jml_negatif = len(df_viz[df_viz['Label_Sentimen'] == 'Negatif'])

            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-card kpi-total">
                    <p class="kpi-title">Total Ulasan Diproses</p>
                    <p class="kpi-value" style="color:#0F172A;">{total_ulasan:,}</p>
                </div>
                <div class="kpi-card kpi-positif">
                    <p class="kpi-title" style="color:#4E79A7;">Sentimen Positif (Biru)</p>
                    <p class="kpi-value" style="color:#4E79A7;">{jml_positif:,} <span style="font-size:0.95rem; font-weight:600;">({round((jml_positif/total_ulasan)*100, 1)}%)</span></p>
                </div>
                <div class="kpi-card kpi-negatif">
                    <p class="kpi-title" style="color:#E15759;">Sentimen Negatif (Merah)</p>
                    <p class="kpi-value" style="color:#E15759;">{jml_negatif:,} <span style="font-size:0.95rem; font-weight:600;">({round((jml_negatif/total_ulasan)*100, 1)}%)</span></p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            color_map = {'Positif': '#4E79A7', 'Negatif': '#E15759'}

            # BARIS 1: 3 CHART BERSAMPINGAN
            col1, col2, col3 = st.columns(3)
            sentimen_counts = df_viz['Label_Sentimen'].value_counts().reset_index()
            sentimen_counts.columns = ['Sentimen', 'Jumlah']

            with col1:
                fig_pie = px.pie(
                    sentimen_counts, values='Jumlah', names='Sentimen',
                    title='Sentimen Keseluruhan',
                    color='Sentimen', color_discrete_map=color_map, hole=0.4
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(margin=dict(t=40, b=20, l=10, r=10), showlegend=True)
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                fig_bar = px.bar(
                    sentimen_counts, x='Sentimen', y='Jumlah',
                    title='Distribusi Jumlah',
                    text='Jumlah', color='Sentimen', color_discrete_map=color_map
                )
                fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
                fig_bar.update_layout(margin=dict(t=40, b=20, l=10, r=10), showlegend=True)
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
                    fig_scatter.update_layout(margin=dict(t=40, b=20, l=10, r=10), showlegend=True)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("Kolom 'Jumlah_Like' tidak ditemukan di dalam data.")

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
                        title="Kecepatan Ulasan Peta Panas"
                    )
                    fig_hm.update_traces(
                        hovertemplate='<b>Hari:</b> %{y}<br><b>Jam:</b> %{x}:00<br><b>Jumlah Ulasan Masuk:</b> %{z}<extra></extra>'
                    )
                    fig_hm.update_layout(margin=dict(t=30, b=20, l=10, r=10), coloraxis_showscale=False)
                    st.plotly_chart(fig_hm, use_container_width=True)
                else:
                    st.info("Kolom 'Tanggal_dan_Jam' tidak ditemukan untuk membuat Heatmap.")

            def buat_wordcloud(teks_gabungan, title, colormap):
                if len(teks_gabungan) > 0:
                    wc = WordCloud(
                        width=500, height=350,
                        background_color='white',
                        colormap=colormap,
                        max_words=80
                    ).generate(teks_gabungan)
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis("off")
                    ax.set_title(title, fontfamily='sans-serif', fontweight='normal', fontsize=14, pad=10, loc='left')
                    fig.tight_layout(pad=1)
                    return fig
                return None

            with col5:
                teks_pos = " ".join(review for review in df_viz[df_viz['Label_Sentimen'] == 'Positif']['Ulasan_Bersih'])
                fig_wc_pos = buat_wordcloud(teks_pos, "Kata Kunci Positif", "Blues")
                if fig_wc_pos:
                    st.pyplot(fig_wc_pos)

            with col6:
                teks_neg = " ".join(review for review in df_viz[df_viz['Label_Sentimen'] == 'Negatif']['Ulasan_Bersih'])
                fig_wc_neg = buat_wordcloud(teks_neg, "Kata Kunci Negatif", "Reds")
                if fig_wc_neg:
                    st.pyplot(fig_wc_neg)
