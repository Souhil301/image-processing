import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Fonction d'étiquetage simplifié
def connected_components_labeling(image):
    labels = np.zeros_like(image, dtype=int)
    label = 1
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] == 255 and labels[i, j] == 0:
                flood_fill(image, labels, i, j, label)
                label += 1
    return labels

def flood_fill(image, labels, i, j, label):
    h, w = image.shape
    stack = [(i, j)]
    while stack:
        x, y = stack.pop()
        if 0 <= x < h and 0 <= y < w:
            if image[x, y] == 255 and labels[x, y] == 0:
                labels[x, y] = label
                stack.extend([(x-1, y), (x+1, y), (x, y-1), (x, y+1)])

def render():
    st.title("Étiquetage des Composantes Connexes")

    st.write("""
    Ce programme permet de segmenter une image binaire en composantes connexes en utilisant une méthode d'étiquetage par parcours en profondeur (flood fill).
    Chargez une image binaire (en noir et blanc) pour tester le processus d'étiquetage.
    """)

    if "uploaded_image" not in st.session_state or st.session_state.uploaded_image is None:
        st.error("Veuillez d’abord charger une image binaire.")
        return

    image = Image.open(st.session_state.uploaded_image).convert("L")
    img_array = np.array(image)
    binary = np.where(img_array > 127, 255, 0).astype(np.uint8)

    # Application de l'étiquetage
    labeled_image = connected_components_labeling(binary)

    # Affichage du résultat
    st.subheader("Image Binarisée")
    st.image(binary, caption="Image Binarisée", use_container_width=True)

    st.subheader("Composantes Connexes Étiquetées")
    st.image(labeled_image, caption="Composantes Connexes", use_container_width=True)

    # Affichage de l'histogramme des composantes connexes
    unique_labels = np.unique(labeled_image)
    unique_labels = unique_labels[unique_labels != 0]  # Ignorer le label '0' (l'arrière-plan)
    st.subheader("Histogramme des Composantes Connexes")
    st.bar_chart({f"Composante {label}": np.sum(labeled_image == label) for label in unique_labels})

