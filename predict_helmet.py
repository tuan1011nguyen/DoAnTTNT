from preproces import preprocess_image
import tensorflow as tf


model_cnn = tf.keras.models.load_model(r'D:\\DoAnTriTueNhanTao\\Model\\model_cnn.h5')
def predict(img):
    img = preprocess_image(img)
    score = model_cnn.predict(img)
    return score