import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import os
from datetime import datetime

# ---------------------------------------------------------
# Load a lightweight model (you can host it in repo as emotion_model.h5)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("emotion_model.h5")
    return model

# class order for this model
CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def predict_emotion(image, model):
    img = image.convert("L").resize((48,48))
    arr = np.array(img) / 255.0
    arr = arr.reshape(1,48,48,1)
    preds = model.predict(arr)
    return CLASS_NAMES[np.argmax(preds)]

def ensure_files():
    os.makedirs("photos", exist_ok=True)
    if not os.path.exists("students.csv"):
        pd.DataFrame(columns=["student_id","name","photo"]).to_csv("students.csv", index=False)
    if not os.path.exists("attendance.csv"):
        pd.DataFrame(columns=["date","student_id","name","emotion","time"]).to_csv("attendance.csv", index=False)

ensure_files()
model = load_model()

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="Emotion Attendance", layout="centered")
st.title("🎓 Emotion-Based Attendance (Light Edition)")

tab1, tab2, tab3 = st.tabs(["🧍 Register", "📸 Attendance", "📊 Dashboard"])

with tab1:
    st.header("Register Student")
    sid = st.text_input("Student ID")
    name = st.text_input("Student Name")
    pic = st.camera_input("Capture Photo")
    if st.button("Register"):
        if sid and name and pic:
            img = Image.open(pic)
            img.save(f"photos/{sid}.jpg")
            df = pd.read_csv("students.csv")
            df.loc[len(df)] = [sid, name, f"photos/{sid}.jpg"]
            df.to_csv("students.csv", index=False)
            st.success(f"{name} registered successfully.")
        else:
            st.warning("Fill all fields and capture photo.")

with tab2:
    st.header("Mark Attendance")
    df = pd.read_csv("students.csv")
    if df.empty:
        st.info("No students yet.")
    else:
        sid = st.selectbox("Select ID", df["student_id"])
        photo = st.camera_input("Capture Photo for Attendance")
        if st.button("Mark Attendance"):
            if photo:
                img = Image.open(photo)
                emotion = predict_emotion(img, model)
                now = datetime.now()
                record = {
                    "date": now.strftime("%Y-%m-%d"),
                    "student_id": sid,
                    "name": df.loc[df["student_id"]==sid,"name"].values[0],
                    "emotion": emotion,
                    "time": now.strftime("%H:%M:%S")
                }
                att = pd.read_csv("attendance.csv")
                att.loc[len(att)] = record
                att.to_csv("attendance.csv", index=False)
                st.success(f"Marked {record['name']} ({emotion}) at {record['time']}")
            else:
                st.warning("Capture an image first.")

with tab3:
    st.header("Attendance Dashboard")
    att = pd.read_csv("attendance.csv")
    if att.empty:
        st.info("No records yet.")
    else:
        st.dataframe(att)
        today = datetime.now().strftime("%Y-%m-%d")
        today_df = att[att["date"]==today]
        if not today_df.empty:
            st.bar_chart(today_df["emotion"].value_counts())
        csv = att.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv, "attendance.csv", "text/csv")
