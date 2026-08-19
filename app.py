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
# ESTILOS — Interfaz de visión artificial premium
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --bg: #060d1a;
        --panel: rgba(15, 26, 46, .55);
        --panel-solid: #0d1730;
        --line: rgba(125, 211, 252, .12);
        --line-strong: rgba(34, 211, 238, .30);
        --text: #eaf3ff;
        --muted: #93a7c7;
        --accent: #22d3ee;
        --accent-2: #3b82f6;
        --valid: #34d399;
        --warn: #fbbf24;
        --danger: #f87171;
    }

    .stApp {
        background:
            radial-gradient(1150px 520px at 12% -6%, rgba(59,130,246,.17), transparent 55%),
            radial-gradient(950px 480px at 90% 8%, rgba(34,211,238,.10), transparent 50%),
            radial-gradient(1200px 720px at 50% 112%, rgba(59,130,246,.09), transparent 62%),
            linear-gradient(180deg, #050b16 0%, #060d1a 55%, #050a14 100%);
        color: var(--text);
    }

    .block-container {
        max-width: 1220px;
        padding-top: 2.1rem;
        padding-bottom: 2.4rem;
    }

    [data-testid="stHeader"] { background: transparent; }

    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-thumb { background: rgba(148,163,184,.22); border-radius: 999px; }
    ::-webkit-scrollbar-track { background: transparent; }

    .stApp, .stMarkdown p, .stMarkdown li {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    h1, h2, h3, .card h3, .hero h1 {
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        letter-spacing: -.01em;
    }

    /* ===== HERO ===== */
    .hero {
        position: relative;
        overflow: hidden;
        padding: 40px 44px 36px;
        border: 1px solid var(--line);
        border-radius: 28px;
        background: linear-gradient(135deg, rgba(16,28,52,.86), rgba(9,18,36,.74));
        box-shadow: 0 30px 70px rgba(2,6,14,.5), inset 0 1px 0 rgba(255,255,255,.04);
        margin-bottom: 26px;
    }
    .hero::before {
        content: "";
        position: absolute; top: -150px; right: -130px;
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(34,211,238,.18), transparent 62%);
        pointer-events: none;
    }
    .hero::after {
        content: "";
        position: absolute; bottom: -170px; left: -110px;
        width: 380px; height: 380px;
        background: radial-gradient(circle, rgba(59,130,246,.15), transparent 62%);
        pointer-events: none;
    }
    .hero-top { position: relative; display: flex; align-items: center; gap: 16px; }
    .logo-mark {
        width: 58px; height: 58px; border-radius: 18px;
        display: grid; place-items: center; flex: none;
        background: linear-gradient(135deg, rgba(34,211,238,.22), rgba(59,130,246,.16));
        border: 1px solid var(--line-strong);
        box-shadow: 0 12px 30px rgba(34,211,238,.16);
    }
    .badge {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 8px 14px; border-radius: 999px;
        color: #c5fbff;
        background: rgba(34,211,238,.10);
        border: 1px solid rgba(34,211,238,.28);
        font-size: .76rem; font-weight: 600;
        letter-spacing: .06em; text-transform: uppercase;
    }
    .hero h1 {
        position: relative;
        font-size: clamp(2.3rem, 4.5vw, 3.5rem);
        line-height: 1.02; margin: 22px 0 10px;
        color: #f7fbff; font-weight: 700;
    }
    .hero h1 .accent { color: var(--accent); }
    .hero-sub {
        position: relative;
        color: var(--muted); font-size: 1.12rem; margin: 0;
    }
    .hero-foot {
        position: relative;
        display: flex; flex-wrap: wrap; gap: 8px 20px;
        margin-top: 22px; color: #7f94b4; font-size: .82rem;
    }
    .hero-foot span { display: inline-flex; align-items: center; gap: 7px; }
    .hero-foot .dot { width: 5px; height: 5px; border-radius: 50%; background: var(--accent); }
    /* ===== CARDS / PANELES ===== */
    .card {
        background: linear-gradient(160deg, rgba(18,31,56,.82), rgba(10,20,38,.72));
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 26px;
        box-shadow: 0 24px 54px rgba(2,6,14,.42), inset 0 1px 0 rgba(255,255,255,.03);
        height: 100%;
    }
    .card h3 { color: #f4f9ff; margin-top: 6px; }
    .section-step {
        display: inline-flex; align-items: center; gap: 8px;
        color: var(--accent); font-size: .72rem; font-weight: 700;
        letter-spacing: .08em; text-transform: uppercase;
    }

    /* Uploader */
    [data-testid="stFileUploader"] { padding: 4px; }
    [data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(180deg, rgba(34,211,238,.05), rgba(255,255,255,.01)) !important;
        border: 1.5px dashed rgba(34,211,238,.35) !important;
        border-radius: 20px !important;
        min-height: 250px;
        display: flex; align-items: center; justify-content: center;
        transition: border-color .2s ease, background .2s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(34,211,238,.65) !important;
        background: rgba(34,211,238,.07) !important;
    }

    /* Cámara */
    [data-testid="stCameraInput"] {
        background: linear-gradient(160deg, rgba(18,31,56,.8), rgba(10,20,38,.7));
        border: 1px solid var(--line);
        border-radius: 20px; overflow: hidden; padding: 8px;
    }
    [data-testid="stCameraInputButton"] {
        background: linear-gradient(135deg, #22d3ee, #3b82f6) !important;
        border: none !important; color: #04222b !important;
        border-radius: 12px !important; font-weight: 700;
    }

    /* Radio segmentada */
    [role="radiogroup"] { gap: 8px !important; }
    [role="radiogroup"] > label {
        background: rgba(148,163,184,.05);
        border: 1px solid rgba(148,163,184,.14);
        border-radius: 12px !important;
        padding: 10px 14px;
        transition: all .2s ease;
    }
    [role="radiogroup"] > label:hover { border-color: rgba(34,211,238,.4); }
    [role="radiogroup"] > label > [data-testid="stMarkdownContainer"] + div { display: none; }

    /* Slider y Toggle */
    .stSlider [data-baseweb="slider"] div[role="slider"] { background: #22d3ee !important; }
    .stSlider [data-testid="stSliderThumbValue"] { color: #cfe9ff !important; }
    .stCheckbox label span:first-child { border-radius: 999px !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(8,15,28,.96), rgba(6,12,23,.96));
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
    .side-brand {
        display: inline-flex; align-items: center; gap: 10px;
        font-family: 'Space Grotesk'; font-size: 1.18rem; font-weight: 700; color: #f4f9ff;
    }
    .side-sub { color: var(--muted); font-size: .85rem; margin: 2px 0 20px; }
    .side-title {
        color: var(--muted); font-size: .72rem; font-weight: 700;
        letter-spacing: .08em; text-transform: uppercase; margin: 4px 0 8px;
    }
    .model-chip {
        display: flex; align-items: center; gap: 10px;
        padding: 10px 12px; border-radius: 14px;
        background: rgba(34,211,238,.06); border: 1px solid rgba(34,211,238,.18);
        font-size: .82rem; color: #cfe9ff; margin-bottom: 8px;
    }
    .model-chip .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--valid); box-shadow: 0 0 10px var(--valid); }

    /* ===== RESULTADO ===== */
    .result-class-row { display: flex; align-items: center; gap: 16px; margin: 6px 0 20px; }
    .result-emoji {
        width: 66px; height: 66px; border-radius: 20px; flex: none;
        display: grid; place-items: center; font-size: 2.1rem;
        background: linear-gradient(135deg, rgba(34,211,238,.18), rgba(59,130,246,.12));
        border: 1px solid var(--line-strong);
        box-shadow: 0 12px 28px rgba(34,211,238,.14);
    }
    .result-label { color: var(--muted); font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; font-weight: 700; }
    .result-class { font-family: 'Space Grotesk'; font-size: 2.1rem; font-weight: 700; color: #f7fbff; line-height: 1.05; }
    .conf-line { display: flex; justify-content: space-between; align-items: baseline; width: 100%; }
    .conf-label { color: var(--muted); font-size: .9rem; }
    .conf-value { font-family: 'Space Grotesk'; font-size: 1.9rem; font-weight: 700; color: var(--accent); }
    .gauge { height: 14px; border-radius: 999px; background: rgba(148,163,184,.16); overflow: hidden; margin-top: 8px; }
    .gauge-fill {
        height: 100%; border-radius: 999px;
        background: linear-gradient(90deg, #22d3ee, #3b82f6);
        box-shadow: 0 0 18px rgba(34,211,238,.5);
        transform-origin: left center;
        animation: grow 1s cubic-bezier(.22,.8,.28,1) both;
    }
    @keyframes grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }

    .top3 { margin-top: 24px; }
    .top3-title { color: var(--muted); font-size: .78rem; letter-spacing: .06em; text-transform: uppercase; font-weight: 700; margin-bottom: 12px; }
    .rank {
        display: flex; align-items: center; gap: 12px;
        padding: 11px 14px; margin-bottom: 9px;
        background: rgba(148,163,184,.05);
        border: 1px solid rgba(148,163,184,.12);
        border-radius: 14px;
    }
    .rank-badge { width: 24px; height: 24px; flex: none; border-radius: 8px; display: grid; place-items: center; font-size: .74rem; font-weight: 700; color: #d9e6f8; background: rgba(148,163,184,.16); }
    .rank-name { flex: 1; font-weight: 600; color: #eaf3ff; }
    .rank-pct { font-family: 'Space Grotesk'; font-weight: 700; color: var(--accent); }
    .rank-bar { width: 120px; height: 7px; border-radius: 999px; background: rgba(148,163,184,.16); overflow: hidden; flex: none; }
    .rank-bar > i { display: block; height: 100%; border-radius: 999px; background: linear-gradient(90deg, #22d3ee, #3b82f6); }

    /* Alertas */
    .stAlert { border-radius: 16px !important; padding: 14px 16px !important; border: 1px solid transparent !important; }
    [data-testid="stAlert"] { border-radius: 16px !important; }
    .stAlert [data-testid="stMarkdownContainer"] p { color: var(--text) !important; }

    /* ===== MÉTRICAS INFERIORES ===== */
    .metric {
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 22px 24px;
        background: linear-gradient(160deg, rgba(18,31,56,.74), rgba(10,20,38,.62));
        box-shadow: 0 16px 40px rgba(2,6,14,.34);
        height: 100%;
    }
    .metric .m-top { display: flex; align-items: center; gap: 12px; }
    .metric .m-icon { font-size: 1.5rem; }
    .metric .m-title { font-family: 'Space Grotesk'; font-weight: 700; color: #f4f9ff; font-size: 1rem; }
    .metric .m-tag { color: var(--accent); font-size: .86rem; font-weight: 600; }
    .metric .m-sub { color: var(--muted); font-size: .85rem; margin-top: 8px; line-height: 1.5; }

    .footer { text-align: center; color: #6f83a3; font-size: .8rem; margin-top: 30px; }
    .foot-brand { color: #9fc1ff; font-weight: 500; }

    .image-card {
        border: 1px solid var(--line); border-radius: 20px; overflow: hidden;
        box-shadow: 0 20px 48px rgba(2,6,14,.45); margin-top: 16px;
    }
    .image-card img { display: block; width: 100%; }
    .img-caption { padding: 10px 16px; color: var(--muted); font-size: .8rem; background: rgba(8,15,28,.6); }

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
# CABECERA
# ============================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-top">
            <div class="logo-mark">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#22d3ee" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="11" cy="11" r="7"></circle>
                    <line x1="21" y1="21" x2="16.5" y2="16.5"></line>
                    <circle cx="11" cy="11" r="2.4" fill="#22d3ee" stroke="none"></circle>
                </svg>
            </div>
            <span class="badge">UTH · Computación en la Nube</span>
        </div>
        <h1>CIFAR <span class="accent">Vision</span></h1>
        <p class="hero-sub">Clasificación inteligente de imágenes con CNN</p>
        <div class="hero-foot">
            <span><i class="dot"></i>Red Neuronal Convolucional</span>
            <span><i class="dot"></i>Dataset CIFAR-10</span>
            <span><i class="dot"></i>Deploy en Streamlit Cloud</span>
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
# ENTRADA
# ============================================================
left, right = st.columns([1.05, 1], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="section-step">01 · Captura</span>', unsafe_allow_html=True)
    st.markdown("### Seleccionar imagen")

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
        "<p style='color:var(--muted);font-size:.84rem;line-height:1.5;margin-top:14px;'>"
        "Para mejores resultados, usa una imagen donde el objeto principal sea claramente visible.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="section-step">02 · Resultado</span>', unsafe_allow_html=True)
    st.markdown("### Predicción")

    if image_file is None:
        st.info("Sube una imagen o toma una fotografía para comenzar.")
        st.markdown(
            "<p style='color:var(--muted);font-size:.9rem;line-height:1.6;margin-top:12px;'>"
            "El flujo es sencillo: imagen → preprocesamiento → CNN → "
            "clase predicha + confianza + alternativas.</p>",
            unsafe_allow_html=True,
        )
    else:
        try:
            image = Image.open(io.BytesIO(image_file.getvalue())).convert("RGB")
            st.image(image, caption="Imagen analizada", width="stretch")

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

        except Exception as exc:
            st.error(f"No se pudo procesar la imagen: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# INFORMACIÓN DEL PROYECTO
# ============================================================
st.markdown("")
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