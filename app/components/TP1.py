import streamlit as st 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def manual_grayscale(rgb_img):
    gray = np.zeros((rgb_img.shape[0], rgb_img.shape[1]), dtype=np.uint8)
    for i in range(rgb_img.shape[0]):
        for j in range(rgb_img.shape[1]):
            r, g, b = rgb_img[i, j]
            intensity = int(0.299 * r + 0.587 * g + 0.114 * b)
            gray[i, j] = intensity
    return gray

def calculate_histogram(gray_img):
    hist = np.zeros(256, dtype=int)
    for value in gray_img.flatten():
        hist[value] += 1
    return hist

def normalize_histogram(hist, total_pixels):
    return hist / total_pixels

def cumulative_histogram(hist):
    return np.cumsum(hist)

def render():
    st.title("Calculs d'Histogrammes d'une Image")

    uploaded_file = st.file_uploader("Uploader une image", type=["jpg", "jpeg", "png", "tif", "tiff"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        # Stockage dans la session
        st.session_state.uploaded_image = uploaded_file
        st.session_state.original_image = img_array

        st.subheader("Image Originale")
        st.image(image, caption="Image chargée", use_container_width=True)

        st.subheader("Conversion en niveaux de gris")
        gray_image = manual_grayscale(img_array)
        st.image(gray_image, caption="Image en niveaux de gris", use_container_width=True)

        col1, col2, col3 = st.tabs(["Histogramme", "Histogramme Normalisé", "Histogramme Cumulé"])
        with col1:
            st.subheader("Histogramme")
            hist = calculate_histogram(gray_image)
            fig1, ax1 = plt.subplots()
            ax1.plot(hist, color='black')
            ax1.set_title("Histogramme")
            ax1.set_xlabel("Niveau de gris")
            ax1.set_ylabel("Nombre de pixels")
            st.pyplot(fig1)
        with col2:
            st.subheader("Histogramme Normalisé")
            norm_hist = normalize_histogram(hist, gray_image.size)
            fig2, ax2 = plt.subplots()
            ax2.plot(norm_hist, color='blue')
            ax2.set_title("Histogramme Normalisé")
            ax2.set_xlabel("Niveau de gris")
            ax2.set_ylabel("Probabilité")
            st.pyplot(fig2)

        with col3:
            st.subheader("Histogramme Cumulé")
            cum_hist = cumulative_histogram(hist)
            fig3, ax3 = plt.subplots()
            ax3.plot(cum_hist, color='green')
            ax3.set_title("Histogramme Cumulé")
            ax3.set_xlabel("Niveau de gris")
            ax3.set_ylabel("Somme cumulée")
            st.pyplot(fig3)

        # Enregistrement aussi des versions calculées
        st.session_state.gray_image = gray_image
        st.session_state.histogram = hist
        st.session_state.normalized_histogram = norm_hist
        st.session_state.cumulative_histogram = cum_hist
