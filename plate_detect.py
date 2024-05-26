import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import re
import numpy as np
from preproces import preprocess_plate

def detect_and_display(image_path):
    # Load the YOLO model
    model = YOLO(r'D:\DoAnTriTueNhanTao\Model\plate_model.pt')
    
    # Read the image
    frame = cv2.imread(image_path)
    if frame is None:
        print("Không thể đọc ảnh từ tệp.")
        return ""
    
    frame = preprocess_plate(frame)
    
    # Get predictions from the model
    results = model(frame)
    
    classes = []  # Danh sách các lớp đã nhận diện
    for result in results:
        boxes = result.boxes  # List of detected bounding boxes
        for box in boxes:
            xmin, ymin, xmax, ymax = box.xyxy[0]  # Tọa độ của bounding box
            confidence = box.conf[0]  # Độ tin cậy của dự đoán
            class_id = box.cls[0]  # ID của lớp
            name = model.names[int(class_id)]  # Tên lớp dựa trên ID

            # Vẽ khung chứa và tên lớp lên ảnh
            label = f'{name} {confidence:.2f}'
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            classes.append((name, confidence, (xmin, ymin, xmax, ymax)))

    # Show the resulting image with bounding boxes
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title('YOLOv8 Detection')
    plt.axis('off')
    plt.show()

    # Process detected classes to format the license plate
    bienso = format_license_plate(classes)
    
    return bienso

def format_license_plate(classes):
    bientren = ''
    bienduoi = ''
    tren = []
    duoi = []
    vach = 640 * (1/3)
    
    for i in classes:
        if i[2][1] < vach:
            tren.append(i)
        else:
            duoi.append(i)

    tren.sort(key=lambda x: x[2][0])
    duoi.sort(key=lambda x: x[2][0])

    for i in tren:
        bientren += i[0]
    for i in duoi:
        bienduoi += i[0]

    if bienduoi:
        bienso = bientren + '-' + bienduoi
    else:
        bienso = re.sub(r'([a-zA-Z])(\d)', r'\1-\2', bientren)
    
    return bienso

# img_path = r'D:\DoAnTriTueNhanTao\test\Screenshot 2024-05-26 192312.jpg'
# print(detect_and_display(img_path))
