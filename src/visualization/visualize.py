import matplotlib.pyplot as plt


def save_predictions(images, predictions, categories):
    plt.figure(figsize=(20, 15))
    for i in range(len(images)):
        ax = plt.subplot(3, 5, i+1)
        plt.imshow(images[i])
        plt.axis("off")
        plt.title(categories[predictions[i]])
    plt.savefig('./outputs/predictions.png')


def save_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./outputs/accuracy_history.png')

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./outputs/loss_history.png')
