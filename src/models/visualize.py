import matplotlib.pyplot as plt


def save_predictions(images, predictions, categories, n_cols, scaler=1):
    plt.figure(figsize=(int(n_cols * 4 * scaler),
                        int(5 * scaler * int(len(images)/n_cols))))
    for i in range(len(images)):
        ax = plt.subplot(len(images)//n_cols + 1, n_cols, i+1)
        plt.imshow(images[i])
        plt.axis("off")
        plt.title(categories[predictions[i] - 1])
    plt.savefig("predictions.png")


def save_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("accuracy_history.png")

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("loss_history.png")
