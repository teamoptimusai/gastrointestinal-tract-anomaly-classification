import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

metrics = ["accuracy",
           tf.keras.metrics.Precision(),
           tf.keras.metrics.Recall(),
           tf.keras.metrics.AUC(),
           tf.keras.metrics.TruePositives(),
           tf.keras.metrics.TrueNegatives(),
           tf.keras.metrics.FalsePositives(),
           tf.keras.metrics.FalseNegatives()
           ]


class GI_NETv2():
    def __init__(self, num_categories, img_size=224):
        self.data_augmentation = tf.keras.Sequential(
            [
                layers.experimental.preprocessing.RandomContrast(0.3),
                layers.experimental.preprocessing.RandomFlip("horizontal",
                                                             input_shape=(img_size,
                                                                          img_size,
                                                                          3)),
                layers.experimental.preprocessing.RandomRotation(0.3),
                layers.experimental.preprocessing.RandomZoom(0.3),
            ]
        )
        self.densenet201 = applications.DenseNet201(
            include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3))
        for layer in self.densenet201.layers:
            layer.trainable = True
        inputs = tf.keras.Input(shape=(img_size, img_size, 3))
        augmented = self.data_augmentation(inputs)
        model_input = tf.keras.applications.densenet.preprocess_input(
            augmented)
        self.densenet_output = self.densenet201(model_input)
        x = tf.keras.layers.GlobalAveragePooling2D()(self.densenet_output)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        predictions = tf.keras.layers.Dense(
            num_categories, activation="softmax")(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=predictions)

    def compile(self):
        self.model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(
            lr=0.001, momentum=0.9), metrics=metrics)

    def fit(self, train_ds, val_ds):
        anne = ReduceLROnPlateau(
            monitor='val_accuracy', factor=0.5, patience=3, verbose=1, min_lr=1e-4)
        checkpoint = ModelCheckpoint(
            'model.h5', verbose=1, save_best_only=True)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True)
        history = self.model.fit(train_ds,
                                 epochs=30,
                                 verbose=1,
                                 validation_data=val_ds,
                                 callbacks=[anne, checkpoint, early_stopping])
        return history

    def evaluate(self, val_ds):
        self.model.evaluate(val_ds)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, save_dir):
        self.model.save(save_dir)
