import streamlit as st
import cv2
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from preproces import return_x_y_z  # Assuming this function extracts sub-images based on bounding boxes
from predict_helmet import predict  # Assuming this function predicts helmet usage
from plate_detect import detect_and_display  # Assuming this function detects and returns the license plate number

# Load models
model = YOLO(r'D:\DoAnTriTueNhanTao\Model\new.pt')
model_cnn = tf.keras.models.load_model(r'D:\DoAnTriTueNhanTao\Model\helmet_detector_cnn.h5')

# Set up output directories
output_plate_path = 'output_path_plate'
output_path_plate = os.path.join(output_plate_path, 'output.png')
output_dir = "output_images"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(output_plate_path):
    os.makedirs(output_plate_path)

# Streamlit app
st.title("Ứng dụng nhận diện đối tượng đi xe máy không đội mũ bảo hiểm và lưu lại biển số xe")
uploaded_file = st.file_uploader("Chọn một file ảnh", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    frame = np.array(image)
    labels = ['head', 'p', 'pb']
    resized_frame = cv2.resize(frame, (850, 1250))
    st.image(resized_frame, caption='Detected Object', use_column_width=True)
    results = model.predict(resized_frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = int(box.cls[0])
            if labels[label] == 'p':
                rs_plate = return_x_y_z(resized_frame, x1, x2, y1, y2)
                rs_plate = cv2.resize(rs_plate, (300, 600))
                cv2.imwrite(output_path_plate, rs_plate)
                label_plate = detect_and_display(output_path_plate)
                cv2.putText(resized_frame, f'{label_plate}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (235, 52, 52), 5)
            elif conf >= 0.1 and labels[label] == 'head':
                head_img = return_x_y_z(resized_frame, x1, x2, y1, y2)
                score = predict(head_img)
                if score[0][0] > 0.8:
                    cv2.putText(resized_frame, 'Co doi mu', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (235, 52, 52), 5)
                else:
                    cv2.putText(resized_frame, 'Khong doi mu', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (3, 12, 3), 5)
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    st.image(resized_frame, caption='YOLOv8 Image Processing')
    output_path = os.path.join(output_dir, 'output.png')
    cv2.imwrite(output_path, resized_frame)
    st.markdown(f"[Tải xuống ảnh đã xử lý](./{output_path})")
else:
    st.write("Vui lòng chọn một file ảnh để tải lên.")
