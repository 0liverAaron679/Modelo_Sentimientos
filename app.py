from datetime import datetime
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from transformers import pipeline

# Configuración inicial de la página
st.set_page_config(
    page_title="Analizador de Sentimientos y Emociones",
    page_icon="📊",
    layout="wide"
)

# Inicialización de estado
if "historial" not in st.session_state:
    st.session_state.historial = []

# ----------------------------------------------------
# CARGA DE MODELOS (Sentimiento y Emoción)
# ----------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelos de IA...")
def cargar_modelos():
    device = 0 if torch.cuda.is_available() else -1
    
    # 1. Modelo de Sentimiento (Positivo, Neutro, Negativo)
    clf_sentimiento = pipeline(
        "sentiment-analysis",
        model="pysentimiento/robertuito-sentiment-analysis",
        truncation=True,
        max_length=128,
        device=device,
        batch_size=1
    )
    
    # 2. Modelo de Emociones (Alegría, Tristeza, Enojo, Miedo, Sorpresa, Asco, Neutro)
    clf_emocion = pipeline(
        "text-classification",
        model="pysentimiento/robertuito-emotion-analysis",
        truncation=True,
        max_length=128,
        device=device,
        batch_size=1
    )
    
    return clf_sentimiento, clf_emocion

clasificador_sent, clasificador_emo = cargar_modelos()

# Mapeos a español y emojis
MAPA_SENTIMIENTOS = {
    "POS": ("Positivo", "😀", "#2ecc71"),
    "POSITIVE": ("Positivo", "😀", "#2ecc71"),
    "NEU": ("Neutro", "😐", "#95a5a6"),
    "NEUTRAL": ("Neutro", "😐", "#95a5a6"),
    "NEG": ("Negativo", "😠", "#e74c3c"),
    "NEGATIVE": ("Negativo", "😠", "#e74c3c")
}

MAPA_EMOCIONES = {
    "JOY": ("Felicidad / Alegría", "😊"),
    "SADNESS": ("Tristeza", "😢"),
    "ANGER": ("Enojo / Ira", "😡"),
    "FEAR": ("Miedo / Temor", "😨"),
    "SURPRISE": ("Sorpresa", "😲"),
    "DISGUST": ("Asco / Desagrado", "🤢"),
    "OTHERS": ("Neutro / Sin emoción clara", "😶")
}

# Sidebar de opciones
with st.sidebar:
    st.header("⚙️ Opciones")
    if st.button("🗑️ Limpiar historial", use_container_width=True):
        st.session_state.historial = []
        st.rerun()

# ----------------------------------------------------
# 1. TÍTULO Y DESCRIPCIÓN
# ----------------------------------------------------
st.title("💬 Analizador de Sentimientos y Emociones en Tiempo Real")
st.markdown("""
Esta herramienta utiliza arquitecturas **Transformer (`RoBERTuito`)**
para clasificar textos en español evaluando tanto el **sentimiento general** (Positivo, Neutro, Negativo) 
como la **emoción específica** detectada (Felicidad, Tristeza, Enojo, Miedo, etc.).
""")

# ----------------------------------------------------
# 2. ENTRADA DE USUARIO
# ----------------------------------------------------
texto = st.text_area(
    "Ingresa un comentario o tweet para analizar:",
    value=st.session_state.get("texto", ""),
    height=110,
    placeholder="Ej: Me da mucha rabia que el paquete haya llegado roto y con retraso..."
)

# ----------------------------------------------------
# 3. INFERENCIA EN TIEMPO REAL
# ----------------------------------------------------
if st.button("🔍 Analizar texto", type="primary", use_container_width=True):
    texto_limpio = texto.strip()
    if not texto_limpio:
        st.warning("Por favor, ingresa un texto válido antes de analizar.")
    else:
        try:
            with torch.no_grad():
                res_sent = clasificador_sent(texto_limpio)[0]
                res_emo = clasificador_emo(texto_limpio)[0]

            # Procesar Sentimiento
            label_sent_raw = res_sent["label"].upper()
            score_sent = res_sent["score"]
            sentimiento, emoji_sent, color_sent = MAPA_SENTIMIENTOS.get(label_sent_raw, (label_sent_raw, "❓", "#7f8c8d"))

            # Procesar Emoción
            label_emo_raw = res_emo["label"].upper()
            score_emo = res_emo["score"]
            emocion, emoji_emo = MAPA_EMOCIONES.get(label_emo_raw, (label_emo_raw, "❓"))

            # Guardar resultados
            st.session_state.historial.append({
                "fecha": datetime.now().strftime("%H:%M:%S"),
                "texto": texto_limpio,
                "sentimiento": sentimiento,
                "prob_sentimiento": score_sent,
                "emocion": emocion,
                "prob_emocion": score_emo
            })

            # Métricas inmediatas
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sentimiento", f"{emoji_sent} {sentimiento}")
            c2.metric("Prob. Sentimiento", f"{score_sent:.1%}")
            c3.metric("Emoción Específica", f"{emoji_emo} {emocion}")
            c4.metric("Prob. Emoción", f"{score_emo:.1%}")

            st.progress(score_sent, text=f"Confianza de Sentimiento: {score_sent:.1%}")
            st.progress(score_emo, text=f"Confianza de Emoción: {score_emo:.1%}")

        except Exception as e:
            st.error(f"Error durante el análisis del texto: {e}")

# ----------------------------------------------------
# 4. VISUALIZACIÓN Y RESUMEN (Gráficos Plotly)
# ----------------------------------------------------
if st.session_state.historial:
    st.divider()
    st.subheader("📊 Visualización y Análisis Acumulado")

    df = pd.DataFrame(st.session_state.historial)

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Total de Análisis", len(df))
    col_m2.metric("Sentimiento Predominante", df["sentimiento"].mode()[0])
    col_m3.metric("Emoción Predominante", df["emocion"].mode()[0])

    g1, g2 = st.columns(2)

    with g1:
        # Gráfico de Sentimientos
        df_sent = df["sentimiento"].value_counts().reset_index()
        df_sent.columns = ["Sentimiento", "Cantidad"]
        fig_sent = px.pie(
            df_sent,
            names="Sentimiento",
            values="Cantidad",
            title="Distribución de Sentimientos General",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    with g2:
        # Gráfico de Emociones
        df_emo = df["emocion"].value_counts().reset_index()
        df_emo.columns = ["Emoción", "Cantidad"]
        fig_emo = px.bar(
            df_emo,
            x="Emoción",
            y="Cantidad",
            title="Distribución de Emociones Específicas",
            color="Emoción",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_emo, use_container_width=True)

    # Historial reciente
    with st.expander("📜 Ver historial completo de análisis"):
        for item in reversed(st.session_state.historial[-10:]):
            with st.container(border=True):
                st.markdown(
                    f"**Sentimiento:** {item['sentimiento']} (`{item['prob_sentimiento']:.1%}`) | "
                    f"**Emoción:** {item['emocion']} (`{item['prob_emocion']:.1%}`) | `{item['fecha']}`"
                )
                st.write(item["texto"])

    st.download_button(
        "⬇️ Descargar Historial Completo (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        "historial_sentimientos_emociones.csv",
        "text/csv",
        use_container_width=True
    )