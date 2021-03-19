import tensorflow as tf
import glob
import numpy as np
import cv2
import argparse
from visualize import save_predictions

categories_dict = {
    "kvasir-capsule": ['Ampulla_of_vater',
                       'Angiectasia',
                       'Blood_fresh',
                       'Blood_hematin',
                       'Erosion',
                       'Erythema',
                       'Foreign_body',
                       'Ileocecal_valve',
                       'Lymphangiectasia',
                       'Normal_clean_mucosa',
                       'Polyp',
                       'Pylorus',
                       'Reduced_mucosal_view',
                       'Ulcer'],
    "kvasir": ['dyed-lifted-polyps',
               'dyed-resection-margins',
               'esophagitis',
               'normal-cecum',
               'normal-pylorus',
               'normal-z-line',
               'polyps',
               'ulcerative-colitis'],
    "hyper-kvasir": ['barretts',
                     'barretts-short-segment',
                     'bbps-0-1',
                     'bbps-2-3',
                     'cecum',
                     'dyed-lifted-polyps',
                     'dyed-resection-margins',
                     'esophagitis-a',
                     'esophagitis-b-d',
                     'hemorrhoids',
                     'ileum',
                     'impacted-stool',
                     'polyps',
                     'pylorus',
                     'retroflex-rectum',
                     'retroflex-stomach',
                     'ulcerative-colitis-grade-0-1',
                     'ulcerative-colitis-grade-1',
                     'ulcerative-colitis-grade-1-2',
                     'ulcerative-colitis-grade-2',
                     'ulcerative-colitis-grade-2-3',
                     'ulcerative-colitis-grade-3',
                     'z-line']
}


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
    parser.add_argument('--save', default=True, type=bool,
                        help="Do you want to save the outputs to a '.txt' file? (default True")
    parser.add_argument('--ncol', default=5, type=int,
                        help="Number of columns in the saved image")
    parser.add_argument('--scaler', default=1.0, type=float,
                        help="Scaling factor for the Prediction image (default 1)")
    args = parser.parse_args()

    for dataset_name in ["kvasir-capsule", "hyper-kvasir", "kvasir"]:
        if dataset_name in args.modeldir.lower():
            print("======== Chose {dataset_name} as the dataset ========")
            categories = categories_dict[dataset_name]
            break

    images, predictions = predict_model(
        args.modeldir, args.imagedir, args.image, args.imgsize)

    np.savetxt("predictions.csv", predictions, delimiter=",")

    predictions_argmax = tf.argmax(predictions, axis=-1).numpy().tolist()

    if args.save:
        save_predictions(images, predictions_argmax,
                         categories, args.ncol, args.scaler)
