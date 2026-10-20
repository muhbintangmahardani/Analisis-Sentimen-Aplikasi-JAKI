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
    page_title="Sentimen JAKI | Pemprov DKI Jakarta",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================================
# 2. CUSTOM UI/UX CSS TEMA RESMI PORTAL JAKI (MODERN GOV-TECH)
# =====================================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Terapkan Font Resmi DKI Jakarta */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: #F8FAFC;
        color: #1E293B;
    }

    /* Hilangkan padding atas berlebih bawaan Streamlit */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1280px !important;
    }

    /* Top Bar & Hero Landing Page JAKI */
    .jaki-topbar {
        background-color: #0F2557;
        color: #FFFFFF;
        padding: 10px 24px;
        border-radius: 12px 12px 0 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        border-bottom: 2px solid #F37021;
    }
    .jaki-hero {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-top: none;
        border-radius: 0 0 16px 16px;
        padding: 32px 32px;
        margin-bottom: 28px;
        box-shadow: 0 4px 15px -3px rgba(15, 37, 87, 0.05);
    }
    .jaki-badge {
        display: inline-block;
        background-color: #EBF3FA;
        color: #0055B8;
        border: 1px solid #BFDBFE;
        font-size: 0.75rem;
        font-weight: 800;
        padding: 5px 12px;
        border-radius: 6px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 14px;
    }
    .jaki-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.25;
        margin-bottom: 12px;
    }
    .jaki-title span {
        color: #0055B8;
    }
    .jaki-desc {
        font-size: 0.95rem;
        color: #475569;
        line-height: 1.65;
        margin-bottom: 20px;
    }
    .pipeline-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 14px;
        margin-top: 18px;
        padding-top: 18px;
        border-top: 1px solid #E2E8F0;
    }
    .pipeline-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 16px;
    }
    .pipeline-step {
        display: inline-block;
        background-color: #0F2557;
        color: #FFFFFF;
        font-size: 0.7rem;
        font-weight: 800;
        padding: 2px 8px;
        border-radius: 4px;
        margin-bottom: 6px;
    }
    .pipeline-card h4 {
        font-size: 0.88rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0 0 4px 0;
    }
    .pipeline-card p {
        font-size: 0.78rem;
        color: #64748B;
        margin: 0;
        line-height: 1.45;
    }

    /* Kustomisasi Tab Navigasi Streamlit */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #FFFFFF;
        padding: 8px 12px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        padding: 0 20px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.9rem;
        color: #475569;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0055B8 !important;
        color: #FFFFFF !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }

    /* Kustomisasi Tombol Utama */
    div.stButton > button[kind="primary"] {
        background-color: #0055B8 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 700 !important;
        transition: all 0.2s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #0F2557 !important;
    }

    /* Kartu Section & Hasil Prediksi */
    .section-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 24px;
        margin-top: 12px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(15, 37, 87, 0.03);
    }
    .success-box {
        padding: 18px 22px;
        border-radius: 12px;
        background-color: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-left: 6px solid #059669;
        margin-top: 14px;
        color: #065F46;
    }
    .error-box {
        padding: 18px 22px;
        border-radius: 12px;
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 6px solid #DC2626;
        margin-top: 14px;
        color: #991B1B;
    }
    .clean-text-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 16px;
        margin-top: 14px;
        font-size: 0.88rem;
        color: #334155;
    }

    /* Styling Custom KPI Cards di Tab 3 */
    .kpi-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        gap: 20px;
        margin-bottom: 25px;
    }
    .kpi-card {
        flex: 1;
        min-width: 240px;
        padding: 22px;
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        text-align: center;
    }
    .kpi-total {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 6px solid #64748B;
    }
    .kpi-positif {
        background-color: #F0F6FC;
        border: 1px solid #DBEAFE;
        border-left: 6px solid #4E79A7;
    }
    .kpi-negatif {
        background-color: #FDF2F2;
        border: 1px solid #FEE2E2;
        border-left: 6px solid #E15759;
    }
    .kpi-title {
        font-size: 0.95rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin: 0;
        padding-bottom: 6px;
    }
    .kpi-value {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# 3. FUNGSI INISIALISASI SASTRAWI & NLTK (SUDAH DIPERBAIKI)
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

# Kamus Normalisasi
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

# =====================================================================
# 4. FUNGSI PEMBERSIHAN TEKS
# =====================================================================
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

# =====================================================================
# 5. FUNGSI LOAD MODEL SVM & TF-IDF
# =====================================================================
@st.cache_resource
def load_models():
    model = joblib.load('model_svm_jaki.pkl')
    vectorizer = joblib.load('tfidf_jaki.pkl')
    return model, vectorizer

try:
    svm_model, tfidf_vect = load_models()
except Exception:
    st.error("Gagal memuat file .pkl. Pastikan model_svm_jaki.pkl dan tfidf_jaki.pkl ada di folder ini!")
    st.stop()

# =====================================================================
# 6. FUNGSI 4 LAPIS GEMBOK VALIDASI KHUSUS DATASET JAKI (ANTI UMUM/SHOPEE)
# =====================================================================
def validasi_khusus_jaki(df):
    """
    Memastikan HANYA file CSV/Excel ulasan Aplikasi JAKI yang memiliki kolom 'Ulasan'
    yang bisa diproses. Semua dataset lain (Shopee, Tokopedia, Gojek, atau CSV umum)
    langsung ditolak mutlak.
    """
    # GEMBOK 1: Wajib memakai kolom bernama 'Ulasan'
    if 'Ulasan' not in df.columns:
        return False, "❌ Akses Ditolak: File wajib memiliki kolom bernama **'Ulasan'** sesuai format dataset resmi JAKI."

    ulasan_series = df['Ulasan'].dropna().astype(str).str.lower()
    total_baris = len(ulasan_series)

    if total_baris == 0:
        return False, "❌ Akses Ditolak: Kolom 'Ulasan' di dalam file kosong."

    # GEMBOK 2: Tolak mutlak jika terdeteksi kata kunci E-Commerce / Ojol / Marketplace
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
            "(E-Commerce / Marketplace / Ojol seperti Shopee, Kurir, Ongkir, Seller). "
            "Sistem ini dikunci khusus hanya untuk dataset ulasan **Aplikasi JAKI Pemprov DKI Jakarta**."
        )

    # GEMBOK 3: Wajib memuat penyebutan eksplisit identitas 'JAKI' atau fitur resminya
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
            "❌ Akses Ditolak (Bukan Dataset JAKI): File yang Anda unggah memiliki kolom 'Ulasan', "
            f"tetapi isinya tidak teridentifikasi sebagai ulasan Aplikasi JAKI (hanya **{round(rasio_inti * 100, 1)}%** baris "
            "yang menyebut entitas JAKI; minimal wajib **5%**)."
        )

    # GEMBOK 4: Konteks Layanan Publik DKI Jakarta
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
            "❌ Akses Ditolak: Konteks kalimat di dalam kolom 'Ulasan' terlalu umum dan tidak relevan "
            "dengan ekosistem layanan publik JAKI Pemprov DKI Jakarta."
        )

    return True, (
        f"✅ Dataset Resmi JAKI Terverifikasi! Ditemukan **{total_baris:,} baris ulasan** "
        f"({jml_sebut_jaki:,} penyebutan eksplisit JAKI & {jml_konteks_dki:,} konteks layanan publik DKI)."
    )

# =====================================================================
# 7. LANDING HERO SECTION (DESAIN PORTAL JAKI PEMPROV DKI)
# =====================================================================
st.markdown("""
    <div class="jaki-topbar">
        <div>🏛️ PORTAL EVALUASI OPINI PUBLIK — JAKARTA SMART CITY (JAKI)</div>
        <div>METODE: SUPPORT VECTOR MACHINE (SVM) &bull; PENELITI: ARTHA NUGRAHA</div>
    </div>
    <div class="jaki-hero">
        <div class="jaki-badge">Jakarta Kini (JAKI) Super-App Analytics</div>
        <div class="jaki-title">
            Dashboard Analisis Sentimen Ulasan Warga pada <span>Aplikasi JAKI</span>
        </div>
        <div class="jaki-desc">
            Platform klasifikasi opini dan keluhan masyarakat DKI Jakarta terhadap layanan aplikasi <strong>JAKI</strong>
            (JakLapor, JakRespons, JakWifi, Pajak, hingga Vaksinasi) secara otomatis menggunakan ekstraksi fitur
            <strong>TF-IDF</strong>, <em>stemming</em> Bahasa Indonesia <strong>Sastrawi</strong>, dan pemisahan
            <em>hyperplane</em> <strong>Support Vector Machine (SVM)</strong>.
        </div>
        <div class="pipeline-grid">
            <div class="pipeline-card">
                <span class="pipeline-step">TAHAP 01</span>
                <h4>Domain Guard & Cleaning</h4>
                <p>Memfilter khusus dataset ulasan JAKI (menolak otomatis CSV e-commerce/aplikasi lain) & normalisasi kata tidak baku.</p>
            </div>
            <div class="pipeline-card">
                <span class="pipeline-step">TAHAP 02</span>
                <h4>Sastrawi Stemming & Stopwords</h4>
                <p>Mengubah kata berimbuhan ke kata dasar Bahasa Indonesia serta mempertahankan kata negasi penting.</p>
            </div>
            <div class="pipeline-card">
                <span class="pipeline-step">TAHAP 03</span>
                <h4>Pembobotan TF-IDF</h4>
                <p>Memetakan frekuensi kata ulasan warga ke dalam matriks numerik berbasis kamus <code>tfidf_jaki.pkl</code>.</p>
            </div>
            <div class="pipeline-card">
                <span class="pipeline-step">TAHAP 04</span>
                <h4>Klasifikasi Margin SVM</h4>
                <p>Menentukan sentimen <strong>Positif</strong> atau <strong>Negatif</strong> beserta tingkat keyakinan model (<code>model_svm_jaki.pkl</code>).</p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

if 'df_hasil' not in st.session_state:
    st.session_state.df_hasil = None

# =====================================================================
# 8. TAB NAVIGASI UTAMA (3 TAB PERSIS SEPERTI SEBELUMNYA)
# =====================================================================
tab1, tab2, tab3 = st.tabs([
    "💬 Analisis Teks Tunggal",
    "📁 Pemrosesan Massal (Khusus CSV JAKI)",
    "📊 Dashboard Visualisasi"
])

# ==========================================
# TAB 1: ANALISIS TEKS TUNGGAL
# ==========================================
with tab1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🔍 Uji Sentimen Ulasan Spesifik")
    st.caption("Masukkan satu kalimat keluhan, saran, atau apresiasi warga terhadap layanan aplikasi JAKI untuk diuji secara langsung.")

    user_input = st.text_area(
        "Ketik keluhan atau ulasan warga secara manual:",
        placeholder="Contoh: Aplikasi JAKI lemot banget pas mau daftar vaksin dan lapor jalan rusak, tolong diperbaiki!",
        height=130
    )

    if st.button("Prediksi Sentimen", type="primary"):
        if user_input.strip() == "":
            st.warning("⚠️ Mohon masukkan teks ulasan terlebih dahulu!")
        else:
            with st.spinner('Menganalisis teks dengan Sastrawi & Model SVM...'):
                teks_bersih = pembersihan_teks(user_input)
                teks_vektor = tfidf_vect.transform([teks_bersih])
                prediction = svm_model.predict(teks_vektor)[0]

                jarak_margin = float(np.ravel(svm_model.decision_function(teks_vektor))[0])
                probabilitas = 1 / (1 + np.exp(-jarak_margin))
                keyakinan = probabilitas if probabilitas > 0.5 else (1 - probabilitas)
                persentase_yakin = round(keyakinan * 100, 1)

                st.markdown(
                    f'<div class="clean-text-box"><strong>🛠️ Hasil Text Preprocessing (Sastrawi):</strong><br>'
                    f'<code>{teks_bersih if teks_bersih else "(Teks habis terfilter stopwords)"}</code></div>',
                    unsafe_allow_html=True
                )

                if prediction == 'Positif':
                    st.markdown(
                        f'<div class="success-box">'
                        f'<h3 style="margin:0; font-size:1.35rem; font-weight:800;">🔵 Sentimen POSITIF (Tingkat Keyakinan: {persentase_yakin}%)</h3>'
                        f'<p style="margin:4px 0 0 0; font-size:0.88rem;">Ulasan menunjukkan kepuasan atau apresiasi warga terhadap layanan Aplikasi JAKI.</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="error-box">'
                        f'<h3 style="margin:0; font-size:1.35rem; font-weight:800;">🔴 Sentimen NEGATIF (Tingkat Keyakinan: {persentase_yakin}%)</h3>'
                        f'<p style="margin:4px 0 0 0; font-size:0.88rem;">Ulasan berisi keluhan kendala teknis (error/lambat) atau ketidakpuasan layanan publik.</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 2: PEMROSESAN MASSAL (DENGAN 4 LAPIS GEMBOK JAKI)
# ==========================================
with tab2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📂 Unggah Dataset Ulasan JAKI (CSV / Excel)")
    st.info(
        "🔒 **Proteksi Dataset Khusus JAKI:** Sistem ini dikunci hanya untuk memproses file CSV/Excel hasil scraping "
        "**Aplikasi JAKI** yang memiliki kolom **`Ulasan`**. Dataset dari aplikasi lain (seperti Shopee, Tokopedia, Gojek, "
        "atau CSV umum) akan **ditolak otomatis** meskipun memiliki kolom `Ulasan`."
    )

    uploaded_file = st.file_uploader("Unggah File Dataset JAKI (.csv atau .xlsx)", type=['csv', 'xlsx'])

    if uploaded_file is None:
        st.session_state.df_hasil = None
    else:
        try:
            if uploaded_file.name.lower().endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # JALANKAN 4 LAPIS GEMBOK VALIDASI KHUSUS JAKI
            is_valid_jaki, pesan_validasi = validasi_khusus_jaki(df)

            if not is_valid_jaki:
                st.error(pesan_validasi)
                st.session_state.df_hasil = None
            else:
                st.success(pesan_validasi)

                if st.button("Mulai Pembersihan & Analisis Massal", type="primary"):
                    with st.status("Memproses & Mengklasifikasikan Data Ulasan JAKI...", expanded=True) as status:
                        df_bersih = df.dropna(subset=['Ulasan']).copy()
                        df_bersih['Ulasan_Bersih'] = df_bersih['Ulasan'].apply(pembersihan_teks)
                        df_bersih = df_bersih[df_bersih['Ulasan_Bersih'].str.strip() != '']

                        vektor_massal = tfidf_vect.transform(df_bersih['Ulasan_Bersih'])
                        df_bersih['Label_Sentimen'] = svm_model.predict(vektor_massal)

                        st.session_state.df_hasil = df_bersih
                        status.update(label="✅ Pemrosesan Massal Selesai!", state="complete", expanded=False)

                    with st.expander("📄 Klik untuk melihat 10 sampel data pertama yang telah diproses", expanded=True):
                        st.dataframe(df_bersih.head(10), use_container_width=True)

                    csv_output = df_bersih.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Unduh Hasil Analisis Lengkap (Hasil_JAKI.csv)",
                        data=csv_output,
                        file_name='Hasil_JAKI.csv',
                        mime='text/csv'
                    )
                    st.success("🎉 Analisis selesai! Silakan klik **Tab 3 (📊 Dashboard Visualisasi)** di atas untuk melihat grafik lengkap.")

        except Exception as e:
            st.error(f"Terjadi kesalahan teknis saat membaca file: {e}")
            st.session_state.df_hasil = None

    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 3: DASHBOARD VISUALISASI EKSKLUSIF
# ==========================================
with tab3:
    if st.session_state.df_hasil is None:
        st.warning("⚠️ Silakan unggah dan proses dataset resmi JAKI di **Tab 2** terlebih dahulu untuk mengaktifkan Dashboard Visualisasi.")
    else:
        df_viz = st.session_state.df_hasil
        total_ulasan = len(df_viz)
        jml_positif = len(df_viz[df_viz['Label_Sentimen'] == 'Positif'])
        jml_negatif = len(df_viz[df_viz['Label_Sentimen'] == 'Negatif'])

        # --- KARTU METRIK KPI ---
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-card kpi-total">
                <p class="kpi-title" style="color:#64748B;">Total Ulasan Diproses</p>
                <p class="kpi-value" style="color:#0F172A;">{total_ulasan:,}</p>
            </div>
            <div class="kpi-card kpi-positif">
                <p class="kpi-title" style="color:#4E79A7;">Sentimen Positif (Biru)</p>
                <p class="kpi-value" style="color:#4E79A7;">{jml_positif:,} <span style="font-size:1.05rem;">({round((jml_positif/total_ulasan)*100, 1)}%)</span></p>
            </div>
            <div class="kpi-card kpi-negatif">
                <p class="kpi-title" style="color:#E15759;">Sentimen Negatif (Merah)</p>
                <p class="kpi-value" style="color:#E15759;">{jml_negatif:,} <span style="font-size:1.05rem;">({round((jml_negatif/total_ulasan)*100, 1)}%)</span></p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        color_map = {'Positif': '#4E79A7', 'Negatif': '#E15759'}

        # ==========================================
        # BARIS 1: 3 CHART BERSAMPINGAN
        # ==========================================
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

        # ==========================================
        # BARIS 2: Heatmap (Kiri), Word Cloud Positif (Tengah), Word Cloud Negatif (Kanan)
        # ==========================================
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
