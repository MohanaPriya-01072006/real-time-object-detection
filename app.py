import streamlit as st
import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np

st.title("Real-Time Object, Hand & Face Detection")

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# Mediapipe initialization
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Webcam input
FRAME_WINDOW = st.image([])  # Streamlit image container
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLO Object detection
    results = yolo_model(frame, stream=True)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls = int(box.cls[0])
            label = yolo_model.names[cls]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Hand detection
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        for landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
            )

    # Face mesh detection
    face_results = face_mesh.process(rgb_frame)
    if face_results.multi_face_landmarks:
        for landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame, landmarks, mp_face.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
            )

    # Display in Streamlit
    FRAME_WINDOW.image(frame, channels="BGR")
