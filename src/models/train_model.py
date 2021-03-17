from src.models import GI_NET
import glob
import tensorflow as tf


def train_model(dataset_dir, img_size = 224, validation_split = 0.2):
    categories = []
    for category in glob.glob("/content/labelled_images/*"):
        categories.append(category[25:])
    print("Catergories : ", categories)

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
    
    model = GI_NET()
    model.compile()
    history = model.fit(train_ds, val_ds)
    return history, model