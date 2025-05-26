import streamlit as st
import numpy as np
from PIL import Image
import math

# Directions pour codage de Freeman (8-connexité)
directions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
              (1, 0), (1, -1), (0, -1), (-1, -1)]

def find_contour(binary):
    """Trouver les coordonnées du contour en 8-connectivité"""
    h, w = binary.shape
    for i in range(h):
        for j in range(w):
            if binary[i, j] == 255:
                return trace_contour(binary, i, j)
    return []

def trace_contour(image, i0, j0):
    i, j = i0, j0
    contour = [(i, j)]
    direction = 7  # initial direction
    while True:
        found = False
        for k in range(8):
            d = (direction + k) % 8
            di, dj = directions[d]
            ni, nj = i + di, j + dj
            if 0 <= ni < image.shape[0] and 0 <= nj < image.shape[1] and image[ni, nj] == 255:
                contour.append((ni, nj))
                i, j = ni, nj
                direction = (d + 5) % 8  # inverse direction
                found = True
                break
        if not found or (i, j) == (i0, j0):
            break
    return contour

def freeman_chain_code(contour):
    code = []
    for k in range(len(contour) - 1):
        p1, p2 = contour[k], contour[k + 1]
        di, dj = p2[0] - p1[0], p2[1] - p1[1]
        for idx, (dx, dy) in enumerate(directions):
            if (di, dj) == (dx, dy):
                code.append(idx)
                break
    return code

# Attributs géométriques
def compute_attributes(region):
    indices = np.argwhere(region == 255)

    # Aire = nombre de pixels blancs
    area = len(indices)

    # Centroïde
    if area == 0:
        return None
    cx = np.mean(indices[:, 1])
    cy = np.mean(indices[:, 0])

    # Périmètre
    perimeter = 0
    for (i, j) in indices:
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < region.shape[0] and 0 <= nj < region.shape[1]:
                if region[ni, nj] == 0:
                    perimeter += 1
                    break

    # Compacité
    if area == 0:
        compactness = 0
    else:
        compactness = (perimeter ** 2) / (4 * math.pi * area)

    # Moments pour l’excentricité
    m00 = area
    m10 = np.sum(indices[:, 1])
    m01 = np.sum(indices[:, 0])
    x̄, ȳ = m10 / m00, m01 / m00

    mu20 = np.sum((indices[:, 1] - x̄) ** 2)
    mu02 = np.sum((indices[:, 0] - ȳ) ** 2)
    mu11 = np.sum((indices[:, 1] - x̄) * (indices[:, 0] - ȳ))

    term1 = (mu20 + mu02) / m00
    term2 = math.sqrt(4 * mu11 ** 2 + (mu20 - mu02) ** 2) / m00
    eccentricity = (term1 + term2) / (term1 - term2 + 1e-5)

    return {
        "Aire": area,
        "Périmètre": perimeter,
        "Compacité": compactness,
        "Centre de gravité": (cx, cy),
        "Excentricité": eccentricity
    }

# Segmentation par seuil (binarisation)
def thresholding(image, threshold=127):
    """Applique un seuil à l'image pour la binariser."""
    return np.where(image > threshold, 255, 0).astype(np.uint8)

def render():
    st.title("TP6 : Codage de Freeman & Attributs géométriques")
    st.write("Analyse des régions segmentées : contour, aire, périmètre, compacité, moments.")

    if "uploaded_image" not in st.session_state or st.session_state.uploaded_image is None:
        st.error("Veuillez d’abord charger une image binaire dans TP5.")
        return

    image = Image.open(st.session_state.uploaded_image).convert("L")
    img_array = np.array(image)

    # Appliquer la binarisation
    binary = thresholding(img_array)

    tab1, tab2, tab3 = st.tabs(["Codage de Freeman", "Attributs", "Histogrammes"])

    with tab1:
        contour = find_contour(binary)
        if contour:
            code = freeman_chain_code(contour)
            st.write("Codage de Freeman :", code)
            canvas = np.zeros_like(binary)
            for (i, j) in contour:
                canvas[i, j] = 255
            st.image(canvas, caption="Contour détecté", use_container_width=True)
        else:
            st.warning("Aucun contour trouvé.")

    with tab2:
        attrs = compute_attributes(binary)
        if attrs:
            st.write("Attributs géométriques :")
            for k, v in attrs.items():
                st.write(f"**{k}** : {v}")
        else:
            st.warning("Aucune région blanche détectée.")

    with tab3:
        areas, compactness = [], []
        labeled = connected_components_labeling(binary)
        n_labels = labeled.max()
        for label in range(1, n_labels + 1):
            region = np.where(labeled == label, 255, 0).astype(np.uint8)
            a = compute_attributes(region)
            if a:
                areas.append(a["Aire"])
                compactness.append(a["Compacité"])
        st.subheader("Histogramme des aires")
        st.bar_chart(areas)
        st.subheader("Histogramme des compacités")
        st.bar_chart(compactness)

# Fonction d’étiquetage simplifiée (copiée depuis TP5)
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
