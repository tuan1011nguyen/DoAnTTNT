import streamlit as st
import cv2
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
import os
from preproces import return_x_y_z
from predict_helmet import predict
from plate_detect import detect_and_display

# Load the YOLO model
model = YOLO(r'D:\DoAnTriTueNhanTao\Model\object_detect.pt')
model_cnn = tf.keras.models.load_model(r'D:\\DoAnTriTueNhanTao\\Model\\model_cnn.h5')  

output_plate_path = 'output_path_plate'
output_path_plate = os.path.join(output_plate_path, 'output.png')
output_dir = "output_images"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

st.title("Ứng dụng nhận diện đối tượng đi xe máy không đội mũ bảo hiểm và lưu lại biển số xe")
uploaded_file = st.file_uploader("Chọn một file video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    video_path = os.path.join("test", uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    output_video_path = 'output_video.mp4'
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    labels = ['head', 'p', 'pb']
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (350, 600))
        results = model.predict(resized_frame)

        for result in results:
            boxes = result.boxes  
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                label = int(box.cls[0])
                if conf >= 0.9 and labels[label] == 'p':
                    rs_plate = return_x_y_z(resized_frame, x1, x2, y1, y2)
                    rs_plate = cv2.resize(rs_plate, (300, 600))
                    cv2.imwrite(output_path_plate, rs_plate)  
                    label_plate = detect_and_display(output_path_plate) 
                    cv2.putText(resized_frame, f'{label_plate}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 52, 52), 2)
                elif conf >= 0.1 and labels[label] == 'head':
                    head_img = return_x_y_z(resized_frame, x1, x2, y1, y2)
                    score = predict(head_img)
                    if score[0][0] > 0.8:
                        cv2.putText(resized_frame, f'Co doi mu', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 52, 52), 2)
                    else:
                        cv2.putText(resized_frame, f'Khong doi mu', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (3, 12, 3), 2)
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Resize back to the original frame size before writing
        output_frame = cv2.resize(resized_frame, (frame_width, frame_height))
        out.write(output_frame)
    
    cap.release()
    out.release()
    
    st.video(output_video_path)
    with open(output_video_path, "rb") as file:
        btn = st.download_button(
            label="Tải xuống video đã xử lý",
            data=file,
            file_name="output_video.mp4",
            mime="video/mp4"
        )
else:
    st.write("Vui lòng chọn một file video để tải lên.")
