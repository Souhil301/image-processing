import streamlit as st
import numpy as np
from PIL import Image

# Calcul manuel d'histogramme
def compute_histogram(image):
    hist = np.zeros(256, dtype=int)
    for value in image.flatten():
        hist[value] += 1
    return hist

# Cumulatif
def compute_cumulative_histogram(hist):
    cum_hist = np.zeros_like(hist)
    cum_sum = 0
    for i in range(len(hist)):
        cum_sum += hist[i]
        cum_hist[i] = cum_sum
    return cum_hist

# Égalisation manuelle
def histogram_equalization_manual(image):
    hist = compute_histogram(image)
    cum_hist = compute_cumulative_histogram(hist)
    total_pixels = image.size
    lut = ((cum_hist - cum_hist.min()) * 255 / (total_pixels - cum_hist.min())).astype(np.uint8)
    return lut[image]

# Inversion
def histogram_inversion(image):
    inverted = 255 - image
    return inverted

# Expansion dynamique (contraste)
def contrast_expansion(image):
    min_val = image.min()
    max_val = image.max()
    if max_val == min_val:
        return image.copy()
    result = ((image - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
    return result

# Translation
def histogram_translation(image, value=50):
    translated = image.astype(int) + value
    translated[translated > 255] = 255
    translated[translated < 0] = 0
    return translated.astype(np.uint8)

# App principale
def render():
    st.title("TP2 : Rehaussement d'image par histogrammes")
    st.write("Ce TP applique plusieurs transformations d'histogrammes à une image mal contrastée.")

    if "uploaded_image" not in st.session_state or st.session_state.uploaded_image is None:
        st.error("Aucune image téléversée. Veuillez téléverser une image dans TP1 d'abord.")
        return

    # Chargement image depuis TP1
    image_pil = Image.open(st.session_state.uploaded_image).convert("L")
    image = np.array(image_pil)

    # Opérations
    translated = histogram_translation(image, value=50)
    inverted = histogram_inversion(image)
    expanded = contrast_expansion(image)
    equalized = histogram_equalization_manual(image)

    # Affichage
    st.subheader("Image originale")
    st.image(image, clamp=True, caption="Image d'entrée (niveaux de gris)", use_container_width=True)

    st.subheader("Transformations")

    col1, col2 = st.columns(2)
    with col1:
        st.image(translated, caption="Translation d'histogramme (+50)", use_container_width=True)
        st.image(inverted, caption="Inversion d'histogramme", use_container_width=True)
    with col2:
        st.image(expanded, caption="Expansion dynamique (contraste)", use_container_width=True)
        st.image(equalized, caption="Égalisation d'histogramme (manuelle)", use_container_width=True)

