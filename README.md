# Coursework 1 - outline plan

1. Speech data collection and labelling
    1. Write short Python script to record audio of fixed length and save to a file
    2. Record ~20 examples of each name for training
    3. Record 10-20 further examples of each name for testing
2. Feature extraction
    1. Implement in Python an algorithm to extract feature vectors from each speech signal
    2. Using filterbank-derived cepstral features
        1. Could start with rectangular channelled filterbank, then move to mel-scaled filterbank
    3. Can augment feature vector with energy and temporal derivatives to increase accuracy
    4. Feature extraction module should take a speech file as input, and output MFCC vectors
3. Acoustic modelling (DNN)
    1. Implement deep neural network using Tensorflow/Keras
    2. Optimise validation accuracy by varying hyperparameters (including changing number of hidden layers and filters)
    3. Use 2D convolutional layers
4. Noise compensation
    1. Must be able to mitigate noise, such as adding spectral subtraction to feature extraction, or training the speech models on noisy speech
    2. Investigate effect of noise and different signal-to-noise ratios
5. Testing/evaluation
    1. Pass new speech files to the training data to let it recognise the speech
    2. Should be able to pass up to 10 separate files, each containing a name, and network should produce a corresponding list of names recognised in those files
    3. Compare classifications to true values, and present confusion matrix of the classifier
    4. Examine the effects of different configurations of the feature extraction, may include:
        1. Number of filterbank channels
        2. Different spacing of the channels
    5. Should be able to explain effects on training/validation data loss and accuracy of changing NN architecture and hyperparameters
    6. May also want to test speech recogniser in noisy conditions and under different SNR ratios
    7. Examine how noise compensation is able to effect performance in noisy conditions
