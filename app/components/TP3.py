import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Fonction de conversion vers un autre espace colorimétrique
def convert_color_space(image, conversion):
    return cv2.cvtColor(image, conversion)

# Quantification uniforme (sans OpenCV)
def uniform_quantization(image, levels):
    factor = 256 // levels
    return (image // factor) * factor

# Quantification par Median-Cut
def median_cut_quantization(image, num_colors):
    pixels = np.float32(image.reshape(-1, 3))
    _, labels, palette = cv2.kmeans(pixels, num_colors, None,
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                    10, cv2.KMEANS_RANDOM_CENTERS)
    quantized = palette[labels.flatten()]
    return quantized.reshape(image.shape).astype(np.uint8)

# Histogramme couleur à partir d'une image (format RGB)
def plot_color_histogram(image, title="Histogramme RGB"):
    color = ('r', 'g', 'b')
    fig, ax = plt.subplots()
    for i, col in enumerate(color):
        hist = np.bincount(image[:, :, i].flatten(), minlength=256)
        ax.plot(hist, color=col, label=col.upper())
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

# Fonction principale Streamlit
def render():
    st.title("🎨 TP3 : Espaces Colorimétriques & Quantification")
    st.write("Affichage et analyse d'image dans plusieurs espaces colorimétriques avec histogrammes et quantification.")

    if "uploaded_image" not in st.session_state or st.session_state.uploaded_image is None:
        st.warning("Veuillez d'abord téléverser une image via TP1.")
        return

    # Chargement et conversion de l'image
    image = Image.open(st.session_state.uploaded_image).convert("RGB")
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # === Espaces Colorimétriques ===
    st.header("🌈 Espaces Colorimétriques")
    hsv = convert_color_space(img_bgr, cv2.COLOR_BGR2HSV)
    lab = convert_color_space(img_bgr, cv2.COLOR_BGR2Lab)
    ycbcr = convert_color_space(img_bgr, cv2.COLOR_BGR2YCrCb)
    cmyk = image.convert("CMYK")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(image, caption="RGB", use_container_width=True)
    with col2:
        st.image(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), caption="HSV", use_container_width=True)
    with col3:
        st.image(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), caption="Lab", use_container_width=True)
    with col4:
        st.image(cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2RGB), caption="YCbCr", use_container_width=True)
    with col5:
        st.image(cmyk, caption="CMYK", use_container_width=True)

    # === Histogramme RGB original ===
    st.header("📊 Histogramme des couleurs (RGB)")
    plot_color_histogram(img_array, title="Histogramme RGB (Original)")

    # === Quantification ===
    st.header("🧮 Quantification des couleurs")
    quant_128 = uniform_quantization(img_array, 128)
    quant_8 = uniform_quantization(img_array, 8)
    median_cut = median_cut_quantization(img_array, 8)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(quant_128, caption="Quantification Uniforme (128)", use_container_width=True)
    with col2:
        st.image(quant_8, caption="Quantification Uniforme (8)", use_container_width=True)
    with col3:
        st.image(median_cut, caption="Median-Cut (8)", use_container_width=True)

    # === Histogramme après quantification ===
    st.header("📊 Histogrammes après quantification")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Uniforme (128)")
        plot_color_histogram(quant_128, title="Hist. Quant. Uniforme (128)")
    with col2:
        st.subheader("Uniforme (8)")
        plot_color_histogram(quant_8, title="Hist. Quant. Uniforme (8)")
    with col3:
        st.subheader("Median-Cut (8)")
        plot_color_histogram(median_cut, title="Hist. Median-Cut (8)")


