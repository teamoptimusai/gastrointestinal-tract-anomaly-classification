import tensorflow as tf
import glob
import  numpy as np
import cv2
import argparse

def predict_model(model_dir, image_dir = None, image_file = None, img_size = 224):
    images = []
    if image_dir:
        image_files = glob.glob(image_dir)
        for image in image_files:
            img = cv2.imread(image)
            img_resized = cv2.resize(img, (img_size, img_size), interpolation = cv2.INTER_AREA)
            images.append(img_resized)
    elif image_file:
        img = cv2.imread(image_file)
        img_resized = cv2.resize(img, (img_size, img_size), interpolation = cv2.INTER_AREA)
        images.append(img_resized)

    model = tf.keras.models.load_model(model_dir)
    predictions = []
    for image in images:
        prediction = model.predict(image)
        predictions.append(prediction)
    return images, predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict Categories for your Images with pretrained Model')
    parser.add_argument('--modeldir', required=True, type=str)
    parser.add_argument('--imagedir', default=None, type=str)
    parser.add_argument('--image', default=None, type=str)
    parser.add_argument('--imgsize', default=224, type=int)
    args = parser.parse_args()

    images, predictions = predict_model(args.modeldir, args.imagedir, args.image, args.imgsize)


