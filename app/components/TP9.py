import streamlit as st
import cv2
import numpy as np

# Fonction pour ouvrir et lire une vidéo
def open_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Impossible d'ouvrir la vidéo.")
        return None

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# Affiche une seule frame choisie
def display_selected_frame(frames, index):
    if frames is None or len(frames) == 0:
        st.error("Aucune image à afficher.")
        return

    frame = frames[index]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, caption=f"Image {index+1}", use_container_width=True)

# Interface Streamlit
def render():
    st.title("Sélectionner une Image d'une Vidéo")

    st.write("Téléchargez une vidéo et choisissez la frame à afficher.")

    uploaded_video = st.file_uploader("Téléchargez une vidéo", type=["mp4", "avi", "mov"], key="uploaded_video")

    if uploaded_video is not None:
        video_path = "/tmp/uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        frames = open_video(video_path)

        if frames:
            frame_index = st.slider("Choisissez une image à afficher :", 0, len(frames)-1, 0)
            display_selected_frame(frames, frame_index)
