import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Histogramme pour niveau de gris
def compute_histogram(image):
    hist = np.zeros(256, dtype=int)
    for value in image.flatten():
        hist[value] += 1
    return hist

# Seuillage global
def global_threshold(image, thresh):
    return np.where(image >= thresh, 255, 0).astype(np.uint8)

# Seuillage d'Otsu manuel
def otsu_threshold(image):
    hist = compute_histogram(image)
    total = image.size
    current_max, threshold = 0, 0
    sum_total, sumB = np.dot(np.arange(256), hist), 0
    wB = 0

    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        between_var = wB * wF * (mB - mF) ** 2
        if between_var > current_max:
            current_max = between_var
            threshold = t
    return global_threshold(image, threshold)

# Gradient de Sobel (support niveaux de gris ou couleur)
def sobel_gradient(image):
    if image.ndim == 3:
        # Image couleur → traitement par canal
        gradients = []
        for c in range(3):
            gradients.append(single_channel_sobel(image[:, :, c]))
        return np.mean(gradients, axis=0).astype(np.uint8)
    else:
        return single_channel_sobel(image)

def single_channel_sobel(channel):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    img_padded = np.pad(channel, ((1, 1), (1, 1)), mode='reflect')
    gx = np.zeros_like(channel, dtype=float)
    gy = np.zeros_like(channel, dtype=float)

    for i in range(channel.shape[0]):
        for j in range(channel.shape[1]):
            region = img_padded[i:i+3, j:j+3]
            gx[i, j] = np.sum(region * sobel_x)
            gy[i, j] = np.sum(region * sobel_y)

    magnitude = np.sqrt(gx**2 + gy**2)
    magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
    return magnitude

# Suppression des maximums locaux (simplifiée)
def non_maximum_suppression(gradient):
    output = np.copy(gradient)
    for i in range(1, gradient.shape[0] - 1):
        for j in range(1, gradient.shape[1] - 1):
            pixel = gradient[i, j]
            if pixel < gradient[i-1, j] or pixel < gradient[i+1, j] or pixel < gradient[i, j-1] or pixel < gradient[i, j+1]:
                output[i, j] = 0
    return output

# Hystérésis (version simplifiée)
def hysteresis_threshold(image, low, high):
    strong = 255
    weak = 50
    result = np.zeros_like(image, dtype=np.uint8)

    strong_i, strong_j = np.where(image >= high)
    weak_i, weak_j = np.where((image >= low) & (image < high))
    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    # Connecter les pixels faibles aux forts
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if result[i, j] == weak:
                if np.any(result[i-1:i+2, j-1:j+2] == strong):
                    result[i, j] = strong
                else:
                    result[i, j] = 0
    return result

# Interface Streamlit
def render():
    st.title("Contours d'une image (TP5)")
    st.write("Détection de contours avec Sobel, suppression des non-max, et hystérésis.")

    if "uploaded_image" not in st.session_state or st.session_state.uploaded_image is None:
        st.error("Veuillez charger une image dans TP1 avant d’utiliser ce module.")
        return

    pil_image = Image.open(st.session_state.uploaded_image)
    image_color = np.array(pil_image)
    image_gray = np.array(pil_image.convert("L"))

    st.image(image_color, caption="Image originale", use_container_width=True)

    method = st.selectbox("Méthode de traitement :", [
        "Gradient (Sobel)",
        "Maximums locaux",
        "Canny simplifié (hystérésis)"
    ])

    gradient = sobel_gradient(image_color)

    if method == "Gradient (Sobel)":
        st.image(gradient, caption="Gradient Sobel |G|", use_container_width=True)

    elif method == "Maximums locaux":
        suppressed = non_maximum_suppression(gradient)
        st.image(suppressed, caption="Après suppression des maximums locaux", use_container_width=True)

    elif method == "Canny simplifié (hystérésis)":
        low = st.slider("Seuil bas :", 0, 255, 50)
        high = st.slider("Seuil haut :", 0, 255, 100)
        edges = hysteresis_threshold(gradient, low, high)
        st.image(edges, caption="Contours détectés (Canny simplifié)", use_container_width=True)
