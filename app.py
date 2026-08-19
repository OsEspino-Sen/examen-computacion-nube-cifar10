import io
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ============================================================
# CONFIGURACIÓN
# ============================================================
st.set_page_config(
    page_title="CIFAR Vision | UTH",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "cifar10_model.keras"

CLASS_NAMES = [
    "Avión",
    "Automóvil",
    "Pájaro",
    "Gato",
    "Ciervo",
    "Perro",
    "Rana",
    "Caballo",
    "Barco",
    "Camión",
]

CLASS_EMOJIS = {
    "Avión": "✈️",
    "Automóvil": "🚗",
    "Pájaro": "🐦",
    "Gato": "🐱",
    "Ciervo": "🦌",
    "Perro": "🐶",
    "Rana": "🐸",
    "Caballo": "🐴",
    "Barco": "🚢",
    "Camión": "🚚",
}

# ============================================================
# ESTILOS — Tema claro profesional
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap');

    :root {
        --bg: #f5f7fb;
        --surface: #ffffff;
        --surface-2: #f8fafc;
        --border: #dfe5ef;
        --border-strong: #c0cde0;
        --text: #14213d;
        --muted: #5b6b85;
        --accent: #2563eb;
        --accent-soft: #eff4ff;
        --accent-border: #dbe7ff;
        --valid: #16a34a;
        --warn: #d97706;
        --danger: #dc2626;
        --shadow: 0 1px 2px rgba(16, 24, 40, .05), 0 8px 24px rgba(16, 24, 40, .06);
    }

    .stApp {
        background:
            radial-gradient(900px 400px at 50% -6%, rgba(37, 99, 235, .05), transparent 60%),
            var(--bg);
        color: var(--text);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.8rem;
        padding-bottom: 2.4rem;
    }

    [data-testid="stHeader"] { background: transparent; }

    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 999px; }
    ::-webkit-scrollbar-track { background: transparent; }

    .stApp, .stMarkdown p, .stMarkdown li {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    .stMarkdown p, .stMarkdown li { color: var(--text); }
    h1, h2, h3, .brand-name, .result-class, .metric .m-title, .side-brand, .section-title, .empty-state .es-title {
        font-family: 'Sora', 'Inter', sans-serif;
        letter-spacing: -.01em;
    }
    .conf-value, .rank-pct, .m-tag, .badge, .step-num, .es-flow {
        font-family: 'JetBrains Mono', monospace;
        font-variant-numeric: tabular-nums;
    }

    /* ===== BARRA SUPERIOR ===== */
    .topbar {
        position: relative;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        flex-wrap: wrap;
        padding: 16px 22px;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        box-shadow: var(--shadow);
        margin-bottom: 22px;
    }
    .topbar::before {
        content: "";
        position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #2563eb, #38bdf8);
    }
    .brand { display: flex; align-items: center; gap: 14px; }
    .logo-mark {
        width: 44px; height: 44px; border-radius: 12px;
        display: grid; place-items: center; flex: none;
        background: var(--accent-soft);
        border: 1px solid var(--accent-border);
    }
    .brand-name { font-size: 1.22rem; font-weight: 700; color: var(--text); }
    .brand-name .accent { color: var(--accent); }
    .brand-sub { font-size: .8rem; color: var(--muted); }
    .topbar-right { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .badge {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 6px 12px; border-radius: 999px;
        background: var(--surface-2); border: 1px solid var(--border);
        color: var(--muted); font-size: .72rem; font-weight: 600;
        letter-spacing: .04em; text-transform: uppercase;
    }
    .badge.accent {
        background: var(--accent);
        border-color: var(--accent);
        color: #ffffff;
    }

    /* ===== CABECERA DE SECCIÓN ===== */
    .section-head { display: flex; align-items: center; gap: 14px; margin: 8px 0 18px; }
    .section-title { color: var(--text); font-size: .95rem; font-weight: 600; }
    .section-rule { flex: 1; height: 1px; background: var(--border); }

    /* ===== CABECERA DE PASO (01 / 02) ===== */
    .step-head { display: flex; align-items: center; gap: 14px; margin-bottom: 20px; }
    .step-num {
        width: 38px; height: 38px; flex: none; border-radius: 10px;
        display: grid; place-items: center;
        font-size: .8rem; font-weight: 600; color: #ffffff;
        background: linear-gradient(135deg, #2563eb, #3b82f6);
        box-shadow: 0 4px 12px rgba(37, 99, 235, .25);
    }
    .step-label { color: var(--muted); font-size: .68rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; }
    .step-head h3 { margin: 2px 0 0; font-size: 1.1rem; color: var(--text); }

    /* ===== CARDS / PANELES (contenedor nativo de Streamlit) ===== */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 26px;
        box-shadow: var(--shadow);
        height: 100%;
        transition: border-color .2s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover { border-color: #cfd8e6; }
    [data-testid="stVerticalBlockBorderWrapper"] > [data-testid="stVerticalBlock"] { gap: .4rem; }

    /* Uploader con visor (viewfinder) */
    [data-testid="stFileUploader"] { padding: 2px; }
    [data-testid="stFileUploaderDropzone"] {
        position: relative;
        background: #fbfcfe !important;
        border: 1.5px dashed var(--border-strong) !important;
        border-radius: 14px !important;
        min-height: 200px;
        display: flex; align-items: center; justify-content: center;
        transition: border-color .2s ease, background .2s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--accent) !important;
        background: #f7faff !important;
    }
    [data-testid="stFileUploaderDropzone"]::before {
        content: "";
        position: absolute; inset: 12px; pointer-events: none; opacity: .55;
        background:
            linear-gradient(var(--accent), var(--accent)) top left / 20px 2px,
            linear-gradient(var(--accent), var(--accent)) top left / 2px 20px,
            linear-gradient(var(--accent), var(--accent)) top right / 20px 2px,
            linear-gradient(var(--accent), var(--accent)) top right / 2px 20px,
            linear-gradient(var(--accent), var(--accent)) bottom left / 20px 2px,
            linear-gradient(var(--accent), var(--accent)) bottom left / 2px 20px,
            linear-gradient(var(--accent), var(--accent)) bottom right / 20px 2px,
            linear-gradient(var(--accent), var(--accent)) bottom right / 2px 20px;
        background-repeat: no-repeat;
    }
    [data-testid="stFileUploader"] [data-testid*="baseButton"] {
        background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        padding: 8px 16px !important;
    }

    /* Cámara */
    [data-testid="stCameraInput"] {
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 14px; overflow: hidden; padding: 8px;
    }
    [data-testid="stCameraInputButton"] {
        background: var(--accent) !important;
        border: none !important; color: #ffffff !important;
        border-radius: 10px !important; font-weight: 600;
    }

    /* Radio segmentada */
    [role="radiogroup"] { gap: 8px !important; }
    [role="radiogroup"] > label {
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 12px !important;
        padding: 9px 13px;
        color: var(--text);
        transition: all .18s ease;
    }
    [role="radiogroup"] > label:hover { border-color: var(--accent); }
    [role="radiogroup"] > label:has(input:checked) {
        background: var(--accent-soft);
        border-color: var(--accent);
        color: #1d4ed8;
        font-weight: 600;
    }
    [role="radiogroup"] > label > [data-testid="stMarkdownContainer"] + div { display: none; }

    /* Slider y Toggle */
    .stSlider [data-baseweb="slider"] div[role="slider"] { background: var(--accent) !important; }
    .stSlider [data-testid="stSliderThumbValue"] { color: var(--text) !important; }
    .stCheckbox label span:first-child { border-radius: 999px !important; }
    .stCheckbox label, .stToggle label, .stSlider label { color: var(--text) !important; }
    [data-testid="stToggle"] input:checked + span {
        background: var(--accent) !important;
    }
    [data-testid="stToggle"] input:checked + span span {
        background: #ffffff !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
    [data-testid="stSidebar"] hr { border-color: var(--border); margin: 16px 0; }
    .side-brand {
        display: inline-flex; align-items: center; gap: 10px;
        font-size: 1.05rem; font-weight: 700; color: var(--text);
    }
    .side-sub { color: var(--muted); font-size: .82rem; margin: 2px 0 16px; }
    .side-title {
        color: var(--muted); font-size: .68rem; font-weight: 600;
        letter-spacing: .1em; text-transform: uppercase; margin: 2px 0 8px;
    }
    .model-chip {
        display: flex; align-items: center; gap: 10px;
        padding: 9px 12px; border-radius: 10px;
        background: var(--surface-2); border: 1px solid var(--border);
        font-size: .8rem; color: #44536e; margin-bottom: 8px;
    }
    .model-chip .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--valid); }

    /* ===== ESTADO VACÍO ===== */
    .empty-state {
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 34px 24px;
        background: var(--surface-2);
        text-align: center;
    }
    .empty-state .es-icon { font-size: 1.9rem; }
    .empty-state .es-title { font-weight: 600; color: var(--text); font-size: .95rem; margin: 12px 0 6px; }
    .empty-state .es-sub { color: var(--muted); font-size: .86rem; line-height: 1.55; margin: 0; }
    .empty-state .es-flow {
        display: inline-flex; align-items: center; justify-content: center; flex-wrap: wrap; gap: 8px;
        margin-top: 18px; padding: 8px 14px; border-radius: 999px;
        background: var(--surface); border: 1px solid var(--border);
        color: var(--muted); font-size: .72rem;
    }
    .es-flow .arrow { color: var(--accent); }

    /* ===== VISTA PREVIA ===== */
    .preview-wrap { margin-top: 18px; }
    [data-testid="stImage"] img {
        border-radius: 12px;
        border: 1px solid var(--border);
    }
    .preview-empty {
        margin-top: 18px;
        border: 1px dashed var(--border-strong);
        border-radius: 14px;
        min-height: 200px;
        display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 6px;
        background: #fbfcfe;
        color: var(--muted); text-align: center; padding: 20px;
    }
    .preview-empty .pe-icon { font-size: 1.7rem; opacity: .8; }
    .preview-empty .pe-title { font-weight: 600; color: var(--text); font-size: .92rem; }
    .preview-empty .pe-sub { font-size: .84rem; }

    /* ===== RESULTADO ===== */
    .result-class-row { display: flex; align-items: center; gap: 16px; margin: 6px 0 22px; }
    .result-emoji {
        width: 62px; height: 62px; border-radius: 16px; flex: none;
        display: grid; place-items: center; font-size: 1.9rem;
        background: var(--accent-soft);
        border: 1px solid var(--accent-border);
    }
    .result-label { color: var(--muted); font-size: .68rem; letter-spacing: .12em; text-transform: uppercase; font-weight: 600; }
    .result-class { font-size: 1.9rem; font-weight: 700; color: var(--text); line-height: 1.08; }
    .conf-line { display: flex; justify-content: space-between; align-items: baseline; width: 100%; }
    .conf-label { color: var(--muted); font-size: .86rem; }
    .conf-value { font-size: 1.7rem; font-weight: 600; color: var(--accent); }
    .gauge { height: 10px; border-radius: 999px; background: #e8edf5; overflow: hidden; margin-top: 10px; }
    .gauge-fill {
        height: 100%; border-radius: 999px;
        background: linear-gradient(90deg, #2563eb, #3b82f6);
        transform-origin: left center;
        animation: grow 1s cubic-bezier(.22, .8, .28, 1) both;
    }
    @keyframes grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }

    .top3 { margin-top: 24px; }
    .top3-title { color: var(--muted); font-size: .68rem; letter-spacing: .12em; text-transform: uppercase; font-weight: 600; margin-bottom: 12px; }
    .rank {
        display: flex; align-items: center; gap: 12px;
        padding: 10px 13px; margin-bottom: 8px;
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 12px;
        transition: border-color .18s ease;
    }
    .rank:hover { border-color: #b9c9f0; }
    .rank-badge {
        width: 24px; height: 24px; flex: none; border-radius: 8px;
        display: grid; place-items: center; font-size: .7rem; font-weight: 600;
        color: var(--muted); background: #e6ecf7;
    }
    .rank-name { flex: 1; font-weight: 500; color: var(--text); }
    .rank-pct { font-size: .85rem; font-weight: 600; color: var(--accent); }
    .rank-bar { width: 110px; height: 6px; border-radius: 999px; background: #e8edf5; overflow: hidden; flex: none; }
    .rank-bar > i { display: block; height: 100%; border-radius: 999px; background: linear-gradient(90deg, #2563eb, #3b82f6); }

    /* Alertas */
    .stAlert { border-radius: 12px !important; padding: 13px 15px !important; border: 1px solid transparent !important; }
    [data-testid="stAlert"] { border-radius: 12px !important; }

    /* ===== MÉTRICAS INFERIORES ===== */
    .metric {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: var(--shadow);
        height: 100%;
        transition: border-color .2s ease;
    }
    .metric:hover { border-color: #c9d4e4; }
    .metric .m-top { display: flex; align-items: center; gap: 12px; }
    .metric .m-icon {
        width: 40px; height: 40px; flex: none; border-radius: 10px;
        display: grid; place-items: center; font-size: 1.1rem;
        background: var(--accent-soft); border: 1px solid var(--accent-border);
    }
    .metric .m-title { font-weight: 600; color: var(--text); font-size: .95rem; }
    .metric .m-tag { color: var(--accent); font-size: .74rem; font-weight: 500; }
    .metric .m-sub { color: var(--muted); font-size: .83rem; margin-top: 10px; line-height: 1.55; }

    .footer { text-align: center; color: #94a3b8; font-size: .78rem; margin-top: 32px; }
    .foot-brand { color: var(--accent); font-weight: 500; }

    @media (prefers-reduced-motion: reduce) { .gauge-fill { animation: none; } }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# MODELO
# ============================================================
@st.cache_resource(show_spinner="Cargando modelo de IA...")
def load_model():
    if not MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ============================================================
# BARRA SUPERIOR
# ============================================================
st.markdown(
    """
    <div class="topbar">
        <div class="brand">
            <div class="logo-mark">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="11" cy="11" r="7"></circle>
                    <line x1="21" y1="21" x2="16.5" y2="16.5"></line>
                    <circle cx="11" cy="11" r="2.4" fill="#2563eb" stroke="none"></circle>
                </svg>
            </div>
            <div>
                <div class="brand-name">CIFAR <span class="accent">Vision</span></div>
                <div class="brand-sub">Clasificación de imágenes con CNN · CIFAR-10</div>
            </div>
        </div>
        <div class="topbar-right">
            <span class="badge">UTH · Computación en la Nube</span>
            <span class="badge accent">Streamlit Cloud</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.error(
        "El modelo todavía no está en el proyecto. "
        "Entrena el Notebook de Google Colab y coloca "
        "`cifar10_model.keras` dentro de `models/`."
    )
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="side-brand">🔎 CIFAR Vision</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-sub">Estudio de clasificación de imágenes</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="side-title">Fuente de imagen</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Fuente de imagen",
        ["📁 Subir imagen", "📷 Usar cámara"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown('<div class="side-title">Resultados</div>', unsafe_allow_html=True)
    show_top3 = st.toggle("Mostrar Top-3", value=True)
    threshold = st.slider(
        "Umbral de confianza",
        min_value=0.10,
        max_value=0.95,
        value=0.50,
        step=0.05,
    )

    st.markdown("---")
    st.markdown('<div class="side-title">Modelo</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="model-chip"><span class="dot"></span>CNN convolucional · CIFAR-10</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="model-chip"><span class="dot"></span>Entrada 32 × 32 × 3 · 10 clases</div>',
        unsafe_allow_html=True,
    )

# ============================================================
# ENTRADA Y PREDICCIÓN — dos paneles balanceados
# ============================================================
left, right = st.columns(2, gap="large")

with left:
    with st.container(border=True):
        st.markdown(
            """
            <div class="step-head">
                <div class="step-num">01</div>
                <div>
                    <div class="step-label">Captura</div>
                    <h3>Seleccionar imagen</h3>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        image_file = None
        if mode == "📁 Subir imagen":
            image_file = st.file_uploader(
                "JPG, JPEG o PNG",
                type=["jpg", "jpeg", "png"],
                max_upload_size=20,
                label_visibility="collapsed",
            )
        else:
            image_file = st.camera_input(
                "Toma una fotografía",
                resolution="720p",
            )

        st.markdown(
            "<p style='color:var(--muted);font-size:.86rem;line-height:1.55;margin-top:12px;'>"
            "Para mejores resultados, usa una imagen donde el objeto principal sea claramente visible.</p>",
            unsafe_allow_html=True,
        )

        image = None
        image_error = None
        if image_file is not None:
            try:
                image = Image.open(io.BytesIO(image_file.getvalue())).convert("RGB")
            except Exception as exc:
                image_error = exc

        if image is not None:
            st.image(image, caption="Imagen analizada", width="stretch")
        elif image_file is None:
            st.markdown(
                """
                <div class="preview-empty">
                    <div class="pe-icon">🖼️</div>
                    <div class="pe-title">Vista previa</div>
                    <div class="pe-sub">Tu imagen aparecerá aquí</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with right:
    with st.container(border=True):
        st.markdown(
            """
            <div class="step-head">
                <div class="step-num">02</div>
                <div>
                    <div class="step-label">Resultado</div>
                    <h3>Predicción</h3>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if image is None:
            if image_file is not None:
                st.error(f"No se pudo procesar la imagen: {image_error}")
            else:
                st.markdown(
                    """
                    <div class="empty-state">
                        <div class="es-icon">📊</div>
                        <div class="es-title">Esperando imagen</div>
                        <p class="es-sub">Sube una imagen o toma una fotografía para comenzar.<br>
                        El análisis se mostrará automáticamente aquí.</p>
                        <div class="es-flow">
                            <span>imagen</span>
                            <span class="arrow">→</span>
                            <span>preprocesamiento</span>
                            <span class="arrow">→</span>
                            <span>CNN</span>
                            <span class="arrow">→</span>
                            <span>predicción</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            # CIFAR-10 usa imágenes RGB de 32x32.
            resized = image.resize((32, 32), Image.Resampling.LANCZOS)
            x = np.asarray(resized, dtype=np.float32)
            x = np.expand_dims(x, axis=0)

            probabilities = model.predict(x, verbose=0)[0]
            order = np.argsort(probabilities)[::-1]
            best_idx = int(order[0])
            best_class = CLASS_NAMES[best_idx]
            confidence = float(probabilities[best_idx])

            if confidence >= threshold:
                st.success("Predicción aceptada por el umbral configurado.")
            else:
                st.warning(
                    "Confianza baja. La CNN está obligada a elegir una clase, "
                    "pero esta imagen puede estar fuera de su distribución."
                )

            st.markdown(
                f"""
                <div class="result-class-row">
                    <div class="result-emoji">{CLASS_EMOJIS[best_class]}</div>
                    <div>
                        <div class="result-label">Predicción principal</div>
                        <div class="result-class">{best_class}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="conf-line">
                    <span class="conf-label">Confianza del modelo</span>
                    <span class="conf-value">{confidence:.2%}</span>
                </div>
                <div class="gauge"><div class="gauge-fill" style="width:{confidence * 100:.2f}%"></div></div>
                """,
                unsafe_allow_html=True,
            )

            if show_top3:
                st.markdown(
                    "<div class='top3-title'>Otras posibilidades · Top-3</div>",
                    unsafe_allow_html=True,
                )
                for rank, idx in enumerate(order[:3], start=1):
                    name = CLASS_NAMES[int(idx)]
                    prob = float(probabilities[int(idx)])
                    st.markdown(
                        f"""
                        <div class="rank">
                            <div class="rank-badge">{rank}</div>
                            <div class="rank-name">{CLASS_EMOJIS[name]} {name}</div>
                            <div class="rank-bar"><i style="width:{prob * 100:.2f}%"></i></div>
                            <div class="rank-pct">{prob:.2%}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# ============================================================
# INFORMACIÓN DEL PROYECTO
# ============================================================
st.markdown(
    """
    <div class="section-head">
        <span class="section-title">Sobre el proyecto</span>
        <span class="section-rule"></span>
    </div>
    """,
    unsafe_allow_html=True,
)
info1, info2, info3 = st.columns(3)

metrics = [
    ("🗂️", "Dataset", "CIFAR-10", "10 clases · imágenes RGB de 32 × 32 píxeles · 50k imágenes de entrenamiento"),
    ("🧠", "Modelo", "CNN", "Red neuronal convolucional entrenada en Google Colab con GPU · accuracy 62.18%"),
    ("☁️", "Deploy", "Streamlit Cloud", "Aplicación web desplegada en la nube y accesible desde cualquier navegador"),
]

for col, (icon, title, tag, sub) in zip([info1, info2, info3], metrics):
    with col:
        st.markdown(
            f"""
            <div class="metric">
                <div class="m-top">
                    <span class="m-icon">{icon}</span>
                    <div>
                        <div class="m-title">{title} · <span class="m-tag">{tag}</span></div>
                    </div>
                </div>
                <div class="m-sub">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div class="footer">
        Proyecto académico · <span class="foot-brand">CIFAR Vision</span> · Computación en la Nube · UTH
    </div>
    """,
    unsafe_allow_html=True,
)
