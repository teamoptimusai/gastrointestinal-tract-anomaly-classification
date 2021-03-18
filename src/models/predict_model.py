from src.visualization.visualize import save_predictions
import tensorflow as tf
import glob
import numpy as np
import cv2
import argparse


def predict_model(model_dir, image_dir=None, image_file=None, img_size=224):
    images = []
    if image_dir:
        image_files = glob.glob(image_dir+"/*.jpg")
        for image in image_files:
            img = cv2.imread(image)
            img_resized = cv2.resize(
                img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            images.append(img_resized)
    elif image_file:
        img = cv2.imread(image_file)
        img_resized = cv2.resize(
            img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        images.append(img_resized)

    model = tf.keras.models.load_model(model_dir)
    images = np.array(images)
    predictions = model.predict(images)
    return images, predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict Categories for your Images with pretrained Model')
    parser.add_argument('--modeldir', required=True, type=str,
                        help="Relative path the h5 model (Required)")
    parser.add_argument('--imagedir', default=None, type=str,
                        help="Relative directory of the Images that need to be predicted (Optional)")
    parser.add_argument('--image', default=None, type=str,
                        help="Relative path to the image file (optional)")
    parser.add_argument('--imgsize', default=224, type=int,
                        help="Image size used in the model (default 224px)")
    parser.add_argument('--save', default = True, type=bool, help="Do you want to save the outputs to a '.txt' file? (default True")
    args = parser.parse_args()

    #code to get categories related to the model
    categories = None
    
    images, predictions = predict_model(
        args.modeldir, args.imagedir, args.image, args.imgsize)

    np.savetxt("results/predictions.csv", predictions, delimiter=",")
    
    predictions_argmax = tf.argmax(predictions).numpy().tolist()

    if args.save:
        save_predictions(images, predictions_argmax, categories)