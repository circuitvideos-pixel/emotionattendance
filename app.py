import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
from datetime import datetime
from PIL import Image
from deepface import DeepFace

# -------------------------------
# Setup
# -------------------------------
os.makedirs("photos", exist_ok=True)
if not os.path.exists("students.csv"):
    pd.DataFrame(columns=["student_id", "name", "photo_path"]).to_csv("students.csv", index=False)
if not os.path.exists("attendance.csv"):
    pd.DataFrame(columns=["date", "student_id", "name", "emotion", "time"]).to_csv("attendance.csv", index=False)

# -------------------------------
# Emotion detection using DeepFace
# -------------------------------
def detect_emotion(image):
    try:
        img_array = np.array(image)
        rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        result = DeepFace.analyze(rgb, actions=["emotion"], enforce_detection=False)
        emotion = result[0]["dominant_emotion"] if isinstance(result, list) else result["dominant_emotion"]
        return emotion
    except Exception as e:
        st.warning(f"Emotion detection error: {e}")
        return "Unknown"

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Emotion Attendance", layout="centered")
st.title("🎓 Emotion-Based Attendance System (DeepFace Edition)")

tab1, tab2, tab3 = st.tabs(["🧍 Register Student", "📸 Take Attendance", "📊 Dashboard"])

# -------------------------------
# Registration
# -------------------------------
with tab1:
    st.subheader("Register New Student")
    student_id = st.text_input("Student ID")
    name = st.text_input("Student Name")
    img_file = st.camera_input("Capture Student Photo")

    if st.button("Register"):
        if not (student_id and name and img_file):
            st.warning("Please fill all fields and capture a photo.")
        else:
            img = Image.open(img_file)
            photo_path = f"photos/{student_id}.jpg"
            img.save(photo_path)

            df = pd.read_csv("students.csv")
            if student_id in df["student_id"].astype(str).values:
                st.warning(f"Student ID {student_id} already exists.")
            else:
                new_row = {"student_id": student_id, "name": name, "photo_path": photo_path}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                df.to_csv("students.csv", index=False)
                st.success(f"{name} registered successfully!")

# -------------------------------
# Attendance
# -------------------------------
with tab2:
    st.subheader("Take Attendance")
    students_df = pd.read_csv("students.csv")
    if students_df.empty:
        st.warning("No students registered yet.")
    else:
        selected_id = st.selectbox("Select Student ID", students_df["student_id"])
        student_name = students_df.loc[students_df["student_id"] == selected_id, "name"].values[0]
        img_att = st.camera_input("Capture Attendance Photo")

        if st.button("Mark Attendance"):
            if img_att:
                image = Image.open(img_att)
                emotion = detect_emotion(image)
                now = datetime.now()

                record = {
                    "date": now.strftime("%Y-%m-%d"),
                    "student_id": selected_id,
                    "name": student_name,
                    "emotion": emotion,
                    "time": now.strftime("%H:%M:%S"),
                }

                att_df = pd.read_csv("attendance.csv")
                att_df = pd.concat([att_df, pd.DataFrame([record])], ignore_index=True)
                att_df.to_csv("attendance.csv", index=False)

                st.success(f"Attendance marked for {student_name} ({emotion}) at {record['time']}")
            else:
                st.warning("Please capture an image first.")

# -------------------------------
# Dashboard
# -------------------------------
with tab3:
    st.subheader("📊 Attendance Dashboard")
    if os.path.exists("attendance.csv"):
        att_df = pd.read_csv("attendance.csv")
        if att_df.empty:
            st.info("No attendance records yet.")
        else:
            st.dataframe(att_df)
            today = datetime.now().strftime("%Y-%m-%d")
            today_df = att_df[att_df["date"] == today]
            if not today_df.empty:
                emotion_counts = today_df["emotion"].value_counts()
                st.bar_chart(emotion_counts)
    else:
        st.warning("No attendance file found.")
