# 🎯 Real-Time Concentration Tracker

A small yet powerful project that uses your **webcam** to track and display your **concentration level** in real-time — perfect for **online exams, interviews, or virtual classrooms**.

Built with **OpenCV**, **MediaPipe**, and **NumPy**, this project helped me explore face detection, eye aspect ratio (EAR), and real-time feedback systems.

---

## 📸 Features

- 👁️ Real-time **face and eye detection** using MediaPipe
- 📊 **Smooth concentration bar** (1–100%)
- 👃 Head alignment detection (based on nose landmark)
- 🕒 Live **FPS and timer display**
- 📈 **Final average concentration** score after session

---

## 🧠 How It Works

The concentration score is calculated based on:
- ✅ **Face presence** (20%)
- 👀 **Eye openness (EAR)** (scaled 0–40%)
- 🎯 **Head centered** (based on nose position) (scaled 0–40%)

The output is displayed live through your webcam feed.

---

## 📦 Requirements

Install dependencies using pip:

```bash
pip install opencv-python mediapipe numpy
