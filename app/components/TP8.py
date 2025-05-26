import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Fonction d'étiquetage simplifié (composantes connexes)
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
                # Ajout des voisins (haut, bas, gauche, droite) à la pile
                stack.extend([(x-1, y), (x+1, y), (x, y-1), (x, y+1)])

# Calcul de la taille et du centre de gravité
def compute_region_attributes(labels, image):
    region_attributes = {}
    for label in np.unique(labels):
        if label == 0:  # Ignorer l'arrière-plan
            continue
        region = np.where(labels == label, 255, 0).astype(np.uint8)
        area = np.sum(region == 255)
        # Calcul du centre de gravité
        indices = np.argwhere(region == 255)
        cx = np.mean(indices[:, 1])
        cy = np.mean(indices[:, 0])
        region_attributes[label] = {
            "Taille": area,
            "Centre de gravité": (cx, cy)
        }
    return region_attributes

# Affichage avec Streamlit
def render():
    st.title("Calcul des Tailles et Centres de Gravité des Régions Segmentées")

    st.write("""
    Ce programme calcule la taille des régions segmentées d'une image binaire et les centres de gravité de ces régions.
    Chargez une image binaire pour tester le processus.
    """)

    if "uploaded_image" not in st.session_state or st.session_state.uploaded_image is None:
        st.error("Veuillez d’abord charger une image binaire.")
        return

    image = Image.open(st.session_state.uploaded_image).convert("L")
    img_array = np.array(image)
    binary = np.where(img_array > 127, 255, 0).astype(np.uint8)

    # Application de l'étiquetage
    labeled_image = connected_components_labeling(binary)

    # Calcul des attributs des régions
    region_attributes = compute_region_attributes(labeled_image, binary)

    # Affichage de l'image binaire
    st.subheader("Image Binarisée")
    st.image(binary, caption="Image Binarisée", use_container_width=True)

    # Affichage des tailles et centres de gravité
    st.subheader("Attributs des Régions Segmentées")
    if region_attributes:
        for label, attributes in region_attributes.items():
            st.write(f"**Composante {label}:**")
            st.write(f"  - Taille (Nombre de pixels) : {attributes['Taille']}")
            st.write(f"  - Centre de gravité : {attributes['Centre de gravité']}")
    else:
        st.warning("Aucune région segmentée trouvée.")

    # Affichage de l'image étiquetée
    st.subheader("Image avec Régions Étiquetées")
    st.image(labeled_image, caption="Régions Étiquetées", use_container_width=True)


