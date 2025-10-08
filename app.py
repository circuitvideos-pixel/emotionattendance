import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import os
from datetime import datetime

# --------------------------------------------------------------------
# Minimal 7-class FER-2013 CNN packaged inside the app (no cv2 needed)
# --------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(48,48,1)),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    # random weights just for demo; replace with trained .h5 if available
    return model

CLASS_NAMES = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]

def predict_emotion(image, model):
    img = image.convert("L").resize((48,48))
    arr = np.array(img)/255.0
    arr = arr.reshape(1,48,48,1)
    preds = model(arr, training=False)
    return CLASS_NAMES[int(tf.argmax(preds, axis=1)[0])]

def ensure_csvs():
    os.makedirs("photos", exist_ok=True)
    if not os.path.exists("students.csv"):
        pd.DataFrame(columns=["student_id","name","photo"]).to_csv("students.csv",index=False)
    if not os.path.exists("attendance.csv"):
        pd.DataFrame(columns=["date","student_id","name","emotion","time"]).to_csv("attendance.csv",index=False)

ensure_csvs()
model = load_model()

# --------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------
st.set_page_config(page_title="Emotion Attendance", layout="centered")
st.title("🎓 Emotion-Based Attendance (No-CV2 Build)")

tab1, tab2, tab3 = st.tabs(["🧍 Register","📸 Attendance","📊 Dashboard"])

# ---------------- Register ----------------
with tab1:
    sid = st.text_input("Student ID")
    name = st.text_input("Name")
    img_file = st.camera_input("Capture Photo")
    if st.button("Register"):
        if sid and name and img_file:
            img = Image.open(img_file)
            img.save(f"photos/{sid}.jpg")
            df = pd.read_csv("students.csv")
            df.loc[len(df)] = [sid,name,f"photos/{sid}.jpg"]
            df.to_csv("students.csv",index=False)
            st.success(f"{name} registered.")
        else:
            st.warning("Please enter details and capture photo.")

# ---------------- Attendance ----------------
with tab2:
    df = pd.read_csv("students.csv")
    if df.empty:
        st.info("No students yet.")
    else:
        sid = st.selectbox("Select ID", df["student_id"])
        img_att = st.camera_input("Capture Attendance Photo")
        if st.button("Mark Attendance"):
            if img_att:
                img = Image.open(img_att)
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
                att.to_csv("attendance.csv",index=False)
                st.success(f"{record['name']} marked as {emotion}")
            else:
                st.warning("Capture a photo first.")

# ---------------- Dashboard ----------------
with tab3:
    att = pd.read_csv("attendance.csv")
    if att.empty:
        st.info("No attendance yet.")
    else:
        st.dataframe(att)
        today = datetime.now().strftime("%Y-%m-%d")
        today_df = att[att["date"]==today]
        if not today_df.empty:
            st.bar_chart(today_df["emotion"].value_counts())
        csv = att.to_csv(index=False).encode()
        st.download_button("📥 Download CSV", csv, "attendance.csv", "text/csv")
