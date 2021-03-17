import tensorflow as tf
import glob
import  numpy as np
import cv2

def predict_model(model_dir, image_dir = None, image_file = None, img_size = 224):
    images = []
    if image_dir:
        image_files = glob.glob(image_dir)
        for image in image_files:
            images.append(np.array(cv2.imread(image)))
    elif image_file:
        image = np.array(cv2.imread(image_file))
        images.append(image)

    model = tf.keras.models.load_model(model_dir)
    predictions = []
    for image in images:
        prediction = model.predict(image)
        predictions.append(prediction)
    return images, predictions



