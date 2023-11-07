import os

from keras.src.layers import Conv2D, MaxPooling2D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import itertools
from keras import Sequential
from keras.layers import Dense, Activation, Flatten, Conv1D, InputLayer, MaxPooling1D
from keras.src.utils import to_categorical
from keras.optimizers.legacy import Adam
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def build_layers(n_classes, input_shape, pool_size):
    model = Sequential()

    model.add(InputLayer(input_shape=(input_shape[0], input_shape[1])))
    model.add(Conv1D(32, 3, activation='elu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    return model


def plot_confusion_matrix(cm, classes):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def test_model(model, test_data, test_labels):
    label_encoder = LabelEncoder()
    label_encoder.fit(test_labels)
    test_labels_enc = label_encoder.transform(test_labels)
    test_labels = to_categorical(test_labels_enc)

    predicted_probs = model.predict(test_data, verbose=0)

    predicted = np.argmax(predicted_probs, axis=1)
    actual = np.argmax(test_labels, axis=1)

    accuracy = metrics.accuracy_score(actual, predicted)

    # print(f'Accuracy: {accuracy * 100}%')
    #
    # cm = confusion_matrix(actual, predicted)
    #
    # plt.figure(figsize=(10, 10))
    #
    # sns.set(font_scale=1.4)
    # sns.heatmap(cm, annot=True, cmap=plt.cm.Blues, fmt='g', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    #
    # plt.xlabel('Predicted labels')
    # plt.ylabel('True labels')
    # plt.title('Confusion Matrix')
    #
    # plt.show()

    return accuracy


def train_model(train_data, train_labels, dnn_params, verbose=0):

    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data, train_labels, test_size=0.2, random_state=42
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels)

    train_labels_enc = label_encoder.transform(train_labels)
    val_labels_enc = label_encoder.transform(val_labels)

    train_labels = to_categorical(train_labels_enc)
    val_labels = to_categorical(val_labels_enc)

    model = build_layers(n_classes=len(train_labels[0]), input_shape=train_data[0].shape, pool_size=3)

    optimizer = Adam(learning_rate=dnn_params['learning_rate'])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    if verbose > 0:
        model.summary()

    model.fit(
        train_data,
        train_labels,
        batch_size=dnn_params['batch_size'],
        epochs=dnn_params['epochs'],
        validation_data=(val_data, val_labels),
        verbose=verbose
    )

    # if __name__ == '__main__':
    #     print(model.summary())
    #
    #     model.fit(
    #         train_data,
    #         train_labels,
    #         batch_size=dnn_params['batch_size'],
    #         epochs=dnn_params['epochs'],
    #         validation_data=(val_data, val_labels),
    #     )
    # else:
    #     model.fit(
    #         train_data,
    #         train_labels,
    #         batch_size=dnn_params['batch_size'],
    #         epochs=dnn_params['epochs'],
    #         validation_data=(val_data, val_labels),
    #         verbose=0
    #     )

    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title(f'Accuracy: {dnn_params}')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Validation'], loc='upper left')
    # plt.show()
    #
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title(f'Loss: {dnn_params}')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Validation'], loc='upper left')
    # plt.show()

    return model
