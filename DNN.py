import numpy as np
import glob
import os

import matplotlib.pyplot as plt


import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, InputLayer, MaxPooling2D
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

from pathlib import Path


#Initialise DNN model
def create_model(shape_in, pool_size):
    numClasses = 20
    model = Sequential()
    model.add(InputLayer(input_shape=(shape_in[0], shape_in[1],1)))
    model.add(Conv2D(32,(3,3), activation='elu'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(numClasses))
    model.add(Activation('softmax'))
    return model






# Normalise the size of each mfcc in dataset as part of preprocessing  
def mfcc_norm(data_path):
    data = []
    labels = []
    max_frames = 0
    for mfcc_file in sorted(glob.glob(data_path)):
        mfcc_data = np.load(mfcc_file)
        if(len(mfcc_data[1]) > max_frames):
            max_frames = len(mfcc_data[1])
            print(max_frames)
    
    #max_frames = get_max_frames(data)
    for item in sorted(glob.glob(data_path)):
        #print(item)
        mfcc_data = np.load(item)
        mfcc_data = np.pad(mfcc_data, ((0,0), (0, max_frames-mfcc_data.shape[1])))
        data.append(mfcc_data)
        
        stemFilename = (Path(os.path.basename(item)).stem)
        label = stemFilename.split('-')
        labels.append(label[0])
    return np.array(labels), np.array(data)



#OUTPUT GRAPHS
def output_graphs(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    

def load_weighting(weightingPath):
    model = create_model()
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=0.01))
    model.load_weights(weightingPath)
    return model



def save_weights(model, path):
    model.save_weights(path)
    
    

def main(train_path, test_size, val_size, num_epoch, num_batch_size):

    labels, mfccs = mfcc_norm(train_path + '/*.npy')
    #test_labels, test_mfccs = mfcc_norm(test_path)
    
    LE = LabelEncoder()
    labels = to_categorical(LE.fit_transform(labels))
    
    X_train, X_tmp, y_train, y_tmp = train_test_split(mfccs, labels, test_size=test_size, random_state=6)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=val_size, random_state=6)
    
    model = create_model([len(mfccs[0]), len(mfccs[0][0])], (3,3))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=0.01))
    
    model.summary()
    
    history = model.fit(mfccs, labels, validation_data=(X_val, y_val), batch_size=num_batch_size, epochs=num_epoch, verbose=1)
    
    output_graphs(history)
    
    predicted_probs = model.predict(X_test, verbose=0)
    predicted = np.argmax(predicted_probs, axis=1)
    actual = np.argmax(y_test, axis=1)
    accuracy = metrics.accuracy_score(actual, predicted)
    print(f'Accuracy: {accuracy * 100}%')
    
    confusion_matrix = metrics.confusion_matrix(np.argmax(y_test, axis=1), predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    cm_display.plot()
    
    save_weights(model, './mfccs/model-weights.h5')
    

if __name__ == "__main__":
    
    main('./mfccs/training', 0.8, 0.5, 100, 16)
 
    """
    epoch = [10,20,30,40,50,75,100]
    validation_part = [0.3, 0.4, 0.5, 0.6, 0.7]
    learning_rate = [0.0001,0.001,0.01,0.1,1,10]   
    batch_size = [8,16,32,64]
    convMatrix_size = [(1,1), (3,3), (5,5), (7,7)]
    
    
    """
    
    
    
    
    
    
    