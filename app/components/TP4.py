import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Appliquer une convolution manuelle sur une image niveau de gris
def manual_filter(image, kernel):
    img_padded = np.pad(image, ((1, 1), (1, 1)), mode='reflect')
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = img_padded[i:i+3, j:j+3]
            output[i, j] = np.clip(np.sum(region * kernel), 0, 255)
    return output

# Appliquer une convolution manuelle sur chaque canal d’une image couleur
def filter_color_image(image, kernel):
    channels = cv2.split(image)
    filtered_channels = [manual_filter(ch, kernel) for ch in channels]
    return cv2.merge(filtered_channels)

# Appliquer un filtre médian manuel sur une image en niveaux de gris
def manual_median_filter(image):
    img_padded = np.pad(image, ((1, 1), (1, 1)), mode='reflect')
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = img_padded[i:i+3, j:j+3].flatten()
            output[i, j] = np.median(region)
    return output

# Appliquer un filtre médian manuel sur une image couleur
def median_filter_color(image):
    channels = cv2.split(image)
    filtered_channels = [manual_median_filter(ch) for ch in channels]
    return cv2.merge(filtered_channels)

# Fonction principale Streamlit
def render():
    st.title("🧪 TP4 : Convolution et Filtres Manuels")
    st.write("Ce TP applique des filtres manuels (convolution ou médian) sur une image chargée.")

    # Vérification que l'image est bien disponible
    if "uploaded_image" not in st.session_state or st.session_state.uploaded_image is None:
        st.error("⚠️ Aucune image trouvée. Veuillez d’abord charger une image dans le TP1.")
        return

    # Chargement de l’image
    image = Image.open(st.session_state.uploaded_image)
    img_array = np.array(image)
    is_color = len(img_array.shape) == 3

    # Choix du filtre à appliquer
    filter_type = st.selectbox("🧰 Choisir un filtre :", [
        "Filtre moyenneur simple",
        "Filtre moyenneur pondéré",
        "Filtre Gaussien",
        "Filtre médian",
        "Filtre passe-haut (accentuation des contours)"
    ])
    
    # Choix du type d’image
    apply_to = st.radio("🎯 Appliquer sur :", ["Niveaux de gris", "Couleur"], index=1 if is_color else 0)

    # Conversion en niveaux de gris si nécessaire
    if apply_to == "Niveaux de gris" and is_color:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Sélection du noyau selon le filtre choisi
    kernel = None
    filtered_image = None

    if filter_type == "Filtre moyenneur simple":
        kernel = np.ones((3, 3), np.float32) / 9
    elif filter_type == "Filtre moyenneur pondéré":
        kernel = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=np.float32) / 16
    elif filter_type == "Filtre Gaussien":
        kernel = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=np.float32) / 16  # Identique au pondéré ici
    elif filter_type == "Filtre passe-haut (accentuation des contours)":
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)
    elif filter_type == "Filtre médian":
        if is_color and apply_to == "Couleur":
            filtered_image = median_filter_color(img_array)
        else:
            filtered_image = manual_median_filter(img_array)

    # Application de la convolution si un noyau est défini
    if kernel is not None:
        if is_color and apply_to == "Couleur":
            filtered_image = filter_color_image(img_array, kernel)
        else:
            filtered_image = manual_filter(img_array, kernel)

    # Affichage de l’image filtrée
    st.image(filtered_image, caption=f"🖼️ Image filtrée ({filter_type})", use_container_width=True)
