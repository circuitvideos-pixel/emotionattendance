import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import face_recognition
from fer import FER
from datetime import datetime
from PIL import Image

# -------------------------------
# Setup folders and CSVs
# -------------------------------
os.makedirs("photos", exist_ok=True)
if not os.path.exists("students.csv"):
    pd.DataFrame(columns=["student_id", "name", "embedding_path"]).to_csv("students.csv", index=False)
if not os.path.exists("attendance.csv"):
    pd.DataFrame(columns=["date", "student_id", "name", "emotion", "time"]).to_csv("attendance.csv", index=False)

# FER detector
emotion_detector = FER(mtcnn=True)

# -------------------------------
# Helper functions
# -------------------------------
def save_face_embedding(image, student_id):
    # Detect face and get encoding
    img_array = np.array(image)
    rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_encodings(rgb)
    if faces:
        np.save(f"photos/{student_id}_embedding.npy", faces[0])
        return f"photos/{student_id}_embedding.npy"
    return None

def identify_student(image):
    known = pd.read_csv("students.csv")
    if known.empty:
        return None, None

    img_array = np.array(image)
    rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    if not encodings:
        return None, None
    current_enc = encodings[0]

    for _, row in known.iterrows():
        stored_enc = np.load(row["embedding_path"])
        match = face_recognition.compare_faces([stored_enc], current_enc, tolerance=0.5)[0]
        if match:
            return row["student_id"], row["name"]
    return None, None

def detect_emotion(image):
    img_array = np.array(image)
    rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    result = emotion_detector.detect_emotions(rgb)
    if result:
        emotions = result[0]["emotions"]
        dominant = max(emotions, key=emotions.get)
        return dominant
    return "Unknown"

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Emotion-based Attendance", layout="centered")
st.title("🎓 Classroom Attendance System with Emotion Detection")

tab1, tab2 = st.tabs(["🧍 Register Student", "📸 Take Attendance"])

# -------------------------------
# Tab 1: Registration
# -------------------------------
with tab1:
    st.subheader("Register New Student")
    name = st.text_input("Student Name")
    student_id = st.text_input("Student ID")
    img_file = st.camera_input("Capture Student Face")

    if img_file and name and student_id:
        image = Image.open(img_file)
        embed_path = save_face_embedding(image, student_id)
        if embed_path:
            df = pd.read_csv("students.csv")
            df = pd.concat([df, pd.DataFrame([{"student_id": student_id, "name": name, "embedding_path": embed_path}])], ignore_index=True)
            df.to_csv("students.csv", index=False)
            st.success(f"{name} registered successfully!")
        else:
            st.error("Face not detected. Try again with better lighting.")

# -------------------------------
# Tab 2: Attendance
# -------------------------------
with tab2:
    st.subheader("Take Attendance with Emotion Detection")
    img_att = st.camera_input("Capture Photo for Attendance")

    if img_att:
        image = Image.open(img_att)
        sid, sname = identify_student(image)
        if sid:
            emotion = detect_emotion(image)
            now = datetime.now()
            record = {
                "date": now.strftime("%Y-%m-%d"),
                "student_id": sid,
                "name": sname,
                "emotion": emotion,
                "time": now.strftime("%H:%M:%S")
            }
            df = pd.read_csv("attendance.csv")
            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
            df.to_csv("attendance.csv", index=False)
            st.success(f"Attendance marked for {sname} ({emotion}) at {record['time']}")
        else:
            st.warning("Student not recognized. Please register first.")

    if st.button("📄 View Attendance Log"):
        df = pd.read_csv("attendance.csv")
        st.dataframe(df)