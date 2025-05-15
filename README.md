# mood-music-recommender
# 🎵 Mood-Based Music Recommendation System

A real-time music recommendation system that detects the user's emotion using computer vision and recommends songs that align with the detected mood using machine learning.

---

## 📌 Problem Statement

In today's fast-paced digital environment, users often turn to music as an emotional outlet. However, conventional music recommendation systems typically rely on listening history or manual input, ignoring the user's real-time emotional state. This leads to recommendations that may not align with the listener's current mood, reducing engagement and satisfaction. The aim of this project is to develop an intelligent music recommendation system through facial expression analysis.


## 💡 Proposed Solution

This system detects emotions such as Happy, Sad, Angry, etc., using facial expression analysis and recommends suitable songs from a curated dataset. It uses a CNN-based emotion detector and a lightweight recommendation engine to enhance user experience through emotional relevance.

---

## 🚀 Features

- 🎥 Real-time emotion detection using webcam
- 🎶 Personalized song recommendations based on mood
- 💡 Easy-to-use interface built with Streamlit
- 📊 Works with local datasets — no need for Spotify API

---

## 🧠 Technologies Used

- **Python**
- **OpenCV** – Real-time face detection
- **TensorFlow/Keras** – CNN for emotion classification
- **Pandas & NumPy** – Data handling
- **scikit-learn** – KNN-based recommendation
- **Streamlit** – Web interface

---

## 🗂️ Folder Structure
- `app.py` - Main Streamlit app
- `emotion_model.h5` - Pretrained CNN model for emotion detection
- `music_data.csv` - Dataset of songs with features like valence, energy, etc.
- `haarcascade_frontalface_default.xml` - Haar cascade for face detection

## ▶️ Running the App

```bash
pip install -r requirements.txt
streamlit run app.py

## 📌 Future Work
-Add multi-user detection
-Integrate live Spotify API
-Expand dataset with regional music

## 📚 Datasets
FER2013 – For training the emotion detection model
https://www.kaggle.com/datasets/msambare/fer2013.  
Please download it manually and place it in the `data/` directory:

Music Dataset – Bollywood/Spotify tracks with valence, energy, tempo, etc.

## 📌 References
-OpenCV Documentation
-Kaggle FER2013 Dataset
-Streamlit Documentation
