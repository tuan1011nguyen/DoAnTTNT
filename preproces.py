import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

def preprocess_image(image):
    image = Image.fromarray(image)
    image = image.resize((224, 224)) 
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0) 
    image = tf.keras.applications.resnet50.preprocess_input(image) 
    return image

def return_x_y_z(resized_frame,x1,x2,y1,y2):
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(resized_frame.shape[1], x2)
    y2 = min(resized_frame.shape[0], y2)
    img_resize = resized_frame[y1:y2, x1:x2]
    return img_resize


def preprocess_plate(img):
    img = cv2.resize(img, (640, 640))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # tăng tương phản
    img_float = np.float32(gray)
    min_val = np.min(img_float)
    max_val = np.max(img_float)
    stretched_img = (img_float - min_val) * (255.0 / (max_val - min_val))
    stretched_img = np.uint8(stretched_img)

    # Áp dụng GaussianBlur giảm nhiễu
    blurred = cv2.GaussianBlur(stretched_img, (5, 5), 0)
    img = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    return img