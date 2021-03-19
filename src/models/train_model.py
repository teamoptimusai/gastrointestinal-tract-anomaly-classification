from visualize import save_history
from model import GI_NETv2
import glob
import tensorflow as tf
import argparse


def train_model(dataset_dir, num_categories, save_dir='./trained_model.h5', img_size=224, validation_split=0.2, interimsavedir='./model.h5'):

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        validation_split=validation_split,
        subset="training",
        label_mode='categorical',
        seed=123,
        image_size=(img_size, img_size),
        batch_size=32)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        validation_split=validation_split,
        subset="validation",
        label_mode='categorical',
        seed=123,
        image_size=(img_size, img_size),
        batch_size=32)

    model = GI_NETv2(num_categories)
    model.compile()
    history = model.fit(train_ds, val_ds, interimsavedir)
    if save_dir:
        model.save(save_dir)
    return history, model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train the Model with yout dataset')
    parser.add_argument('--dataset', required=True, type=str,
                        help="relative/absolute dir path to the dataset")
    parser.add_argument('--categories', required=True,
                        type=int, help="number of categories in the dataset")
    parser.add_argument('--savedir', default='./trained_model.h5', type=str,
                        help="relative/absolute path to the best-model.h5 (end with .h5 file format)")
    parser.add_argument('--interimsavedir', default='./interim-model.h5', type=str,
                        help="relative/absolute path to the interim-model.h5 (end with .h5 file format)")
    parser.add_argument('--imgsize', default=224, type=int,
                        help="image size to crop the images")
    parser.add_argument('--valsplit', default=0.2, type=float,
                        help="validation split ratio (default set to 0.2)")
    args = parser.parse_args()

    history, model = train_model(
        args.dataset, args.categories, args.savedir, args.imgsize, args.valsplit, args.interimsavedir)

    save_history(history)
