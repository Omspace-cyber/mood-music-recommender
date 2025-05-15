import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import os

# -----------------------------
# Load model and encoder
# -----------------------------
model = load_model('emotion_model.h5')
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral']
label_encoder = LabelEncoder()
label_encoder.fit(emotions)

# -----------------------------
# Emotion-to-Music Mapping
# -----------------------------
emotion_to_music = {
    'angry':     {'energy': 0.8, 'danceability': 0.7, 'tempo': 120, 'valence': 0.3},
    'happy':     {'energy': 0.9, 'danceability': 0.9, 'tempo': 130, 'valence': 0.9},
    'sad':       {'energy': 0.3, 'danceability': 0.3, 'tempo': 70,  'valence': 0.2},
    'surprised': {'energy': 0.7, 'danceability': 0.8, 'tempo': 110, 'valence': 0.7},
    'disgust':   {'energy': 0.5, 'danceability': 0.6, 'tempo': 100, 'valence': 0.4},
    'fear':      {'energy': 0.4, 'danceability': 0.5, 'tempo': 90,  'valence': 0.1},
    'neutral':   {'energy': 0.6, 'danceability': 0.6, 'tempo': 110, 'valence': 0.5}
}

# -----------------------------
# Predict Emotion
# -----------------------------
def predict_emotion(face_img):
    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized_face = cv2.resize(gray_face, (48, 48))
    normalized = resized_face.astype('float32') / 255.0
    reshaped = normalized.reshape(1, 48, 48, 1)
    prediction = model.predict(reshaped)
    predicted_label = np.argmax(prediction)
    emotion = label_encoder.inverse_transform([predicted_label])[0]
    print("Prediction probabilities:", prediction)
    print("Predicted emotion:", emotion)
    return emotion

# -----------------------------
# Recommend Songs
# -----------------------------
def recommend_songs(mood, music_df, n=10):
    user_features = np.array([list(emotion_to_music[mood].values())]).reshape(1, -1)
    features = ['danceability', 'energy', 'tempo', 'valence']
    song_features = music_df[features].values

    knn = NearestNeighbors(n_neighbors=n)
    knn.fit(song_features)
    _, indices = knn.kneighbors(user_features)

    return music_df.iloc[indices[0]]

# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.title("🎵 Mood-Based Music Recommender")
    st.write("Detect your emotion from webcam and get song recommendations!")

    # Load music data
    music_df = pd.read_csv('database.csv')
    features = ['danceability', 'energy', 'tempo', 'valence']
    scaler = MinMaxScaler()
    music_df[features] = scaler.fit_transform(music_df[features])

    run = st.button("Capture Mood & Recommend Music")
    FRAME_WINDOW = st.image([])

    if run:
        camera = cv2.VideoCapture(0)
        ret, frame = camera.read()
        camera.release()

        if not ret:
            st.error("Failed to access webcam.")
            return

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Captured Image")

        # Detect face
        cascade_path = 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        face_cascade = cv2.CascadeClassifier(cascade_path)
        # if face_cascade.empty():
        #     raise IOError("Cannot load haarcascade_frontalface_default.xml")
        # img = cv2.imread('path_to_image.jpg')  # or from webcam or video stream
        # if img is None:
        #     raise ValueError("Image not loaded. Check the file path.")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            st.error("No face detected. Please make sure your face is clearly visible to the webcam.")
            return

        (x, y, w, h) = faces[0]
        face_img = frame[y:y+h, x:x+w]

        # Predict emotion
        mood = predict_emotion(face_img)
        st.success(f"🧠 Detected Emotion: **{mood.capitalize()}**")

        # Recommend songs
        recommendations = recommend_songs(mood, music_df)
        st.subheader("🎶 Top Recommended Songs")
        for i, row in recommendations.iterrows():
            st.write(f"**→ {row['name']}** — {row.get('artist', 'Unknown Artist')}")

if __name__ == "__main__":
    main()
