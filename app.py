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
# ESTILOS
# ============================================================
st.markdown(
    """
    <style>
    :root {
        --bg: #07111f;
        --panel: #0e1b2f;
        --panel-2: #111f36;
        --line: #223654;
        --text: #eef5ff;
        --muted: #9fb1c9;
        --accent: #5eead4;
        --accent-2: #60a5fa;
        --danger: #fb7185;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(96,165,250,.16), transparent 27%),
            radial-gradient(circle at 90% 10%, rgba(94,234,212,.12), transparent 25%),
            linear-gradient(180deg, #06101d 0%, #081525 45%, #07111f 100%);
        color: var(--text);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stHeader"] {
        background: rgba(7,17,31,.78);
    }

    .hero {
        padding: 28px 30px;
        border: 1px solid rgba(148,163,184,.18);
        border-radius: 26px;
        background:
            linear-gradient(135deg, rgba(14,27,47,.94), rgba(12,29,51,.76));
        box-shadow: 0 25px 60px rgba(0,0,0,.24);
        margin-bottom: 22px;
    }

    .badge {
        display: inline-block;
        padding: 7px 11px;
        border-radius: 999px;
        color: #d9fffb;
        background: rgba(94,234,212,.10);
        border: 1px solid rgba(94,234,212,.26);
        font-size: .80rem;
        font-weight: 700;
        letter-spacing: .04em;
        text-transform: uppercase;
    }

    .hero h1 {
        font-size: clamp(2rem, 4vw, 3.4rem);
        line-height: 1.04;
        margin: 14px 0 10px;
        color: #f8fbff;
    }

    .hero p {
        color: var(--muted);
        font-size: 1.04rem;
        margin: 0;
        max-width: 820px;
    }

    .card {
        border: 1px solid rgba(148,163,184,.16);
        border-radius: 22px;
        background: linear-gradient(145deg, rgba(14,27,47,.92), rgba(10,22,39,.88));
        padding: 22px;
        box-shadow: 0 18px 45px rgba(0,0,0,.18);
        height: 100%;
    }

    .card h3 {
        margin-top: 0;
        color: #f8fbff;
    }

    .mini {
        color: var(--muted);
        font-size: .92rem;
        line-height: 1.55;
    }

    .result-title {
        font-size: 2rem;
        font-weight: 800;
        color: #f8fbff;
        margin-top: 4px;
    }

    .confidence {
        color: #9ef7eb;
        font-size: 1.05rem;
        font-weight: 700;
    }

    .metric {
        border: 1px solid rgba(96,165,250,.16);
        border-radius: 16px;
        padding: 14px 16px;
        background: rgba(96,165,250,.06);
    }

    .small-note {
        color: #91a6bf;
        font-size: .82rem;
        line-height: 1.5;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(255,255,255,.018);
        border-radius: 18px;
        padding: 8px;
    }

    .footer {
        text-align:center;
        color:#7185a1;
        font-size:.82rem;
        margin-top:28px;
    }
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
        <span class="badge">EXAMEN · COMPUTACIÓN EN LA NUBE · UTH</span>
        <h1>🔎 CIFAR Vision</h1>
        <p>
            Clasificación de imágenes con una red neuronal convolucional (CNN)
            entrenada con CIFAR-10 y desplegada como aplicación web mediante Streamlit.
        </p>
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
    st.markdown("## ⚙️ Control")
    mode = st.radio(
        "Fuente de imagen",
        ["📁 Subir imagen", "📷 Usar cámara"],
        index=0,
    )
    show_top3 = st.toggle("Mostrar Top-3", value=True)
    threshold = st.slider(
        "Umbral de confianza",
        min_value=0.10,
        max_value=0.95,
        value=0.50,
        step=0.05,
    )

    st.markdown("---")
    st.markdown("### 🧠 Modelo")
    st.caption("CNN convolucional · CIFAR-10")
    st.caption("Entrada esperada: 32 × 32 × 3")
    st.caption("10 clases")

# ============================================================
# ENTRADA
# ============================================================
left, right = st.columns([1.05, 1], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 1. Selecciona una imagen")

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
        '<p class="small-note">Consejo: para mejores resultados, utiliza una imagen donde el objeto principal sea claramente visible.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 2. Predicción")

    if image_file is None:
        st.info("Sube una imagen o toma una fotografía para comenzar.")
        st.markdown(
            """
            <div class="mini">
                El flujo es sencillo: imagen → preprocesamiento → CNN → clase
                predicha + confianza + alternativas.
            </div>
            """,
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

            st.markdown(
                f'<div class="result-title">{CLASS_EMOJIS[best_class]} {best_class}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="confidence">Confianza: {confidence:.2%}</div>',
                unsafe_allow_html=True,
            )

            if confidence >= threshold:
                st.success("Predicción aceptada por el umbral configurado.")
            else:
                st.warning(
                    "Confianza baja. La CNN está obligada a elegir una clase, "
                    "pero esta imagen puede estar fuera de su distribución."
                )

            if show_top3:
                st.markdown("#### Otras posibilidades")
                for rank, idx in enumerate(order[:3], start=1):
                    name = CLASS_NAMES[int(idx)]
                    prob = float(probabilities[int(idx)])
                    st.progress(
                        prob,
                        text=f"{rank}. {CLASS_EMOJIS[name]} {name} · {prob:.2%}",
                    )

        except Exception as exc:
            st.error(f"No se pudo procesar la imagen: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# INFORMACIÓN DEL PROYECTO
# ============================================================
st.markdown("")
info1, info2, info3 = st.columns(3)

with info1:
    st.markdown(
        """
        <div class="metric">
            <strong>📦 Dataset</strong><br>
            <span class="mini">CIFAR-10 · 10 clases · imágenes RGB 32×32</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with info2:
    st.markdown(
        """
        <div class="metric">
            <strong>🧠 IA</strong><br>
            <span class="mini">Red neuronal convolucional entrenada en Google Colab</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with info3:
    st.markdown(
        """
        <div class="metric">
            <strong>☁️ Nube</strong><br>
            <span class="mini">Aplicación lista para Streamlit Community Cloud</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="footer">
        Proyecto académico · Computación en la Nube · UTH
    </div>
    """,
    unsafe_allow_html=True,
)
