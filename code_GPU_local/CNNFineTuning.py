import os
import numpy as np
from PIL import Image

from keras.datasets import cifar10
from keras.layers import Dense
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.applications import resnet50

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#import Utils.resnet50 as resnet

#--------------------------------------------------------------------------------------------------------------------------------------------------------------

def loadingImages():
    print("\tLoading CIFAR10 images ...")
    (Xtrain, Ytrain), (Xtest, Ytest) = cifar10.load_data()

    Xtrain, Ytrain = lowSampleDataset(Xtrain, Ytrain)
    Xtest, Ytest = lowSampleDataset(Xtest, Ytest)

    X = []
    for i in range(0, Xtrain.shape[0]):
        X.append(np.array(Image.fromarray(Xtrain[i]).resize(size=(224,224))))
    Xtrain = np.array(X)

    X = []
    for i in range(0, Xtest.shape[0]):
        X.append(np.array(Image.fromarray(Xtest[i]).resize(size=(224,224))))
    Xtest = np.array(X)

    Ytrain = to_categorical(Ytrain)
    Ytest = to_categorical(Ytest)
    
    print('\t\tTraining set shape: ', Xtrain.shape)
    print('\t\tTesting set shape: ', Xtest.shape)
    return Xtrain, Ytrain, Xtest, Ytest

def lowSampleDataset(X, Y):
    perm = np.random.permutation(X.shape[0])
    X = X[perm[0 : (int)(X.shape[0] * (5/100))]]
    Y = Y[perm[0 : (int)(Y.shape[0] * (5/100))]]
    return X, Y

def setModel(numberClasses):
    print("\tLoading the ResNet50-ImageNet model")
    model = resnet.ResNet50(input_shape=(224, 224, 3), weights='imagenet', include_top=True, classes=1000)
    model = Model(inputs=model.input, outputs=model.get_layer(index=-2).output)
    
    print("\tAdding new layers to the model ...")
    newTop = Sequential()
    newTop.add(Dense(numberClasses, activation='softmax', name='fc1000', input_shape=model.output_shape[1:]))
    
    model = Model(inputs=model.input, outputs=newTop(model.output))

    print("\tSet fine-tuning configuration...")
    #for layer in model.layers[:-int(10)]:
    #    layer.trainable = False

    optimizer = SGD(lr=0.01, momentum=0.0001, decay=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    #model.summary()

    return model

def trainingModel(model, X, Y, batchSize, numberEpochs):
    print("\tTraining the model ...")

    batches = list(range(0, len(Y), batchSize))
    perm = np.random.permutation(len(Y))

    errLoss = []
    accLoss = []
    errLoss.append(1)
    accLoss.append(0)

    for e in range(0, numberEpochs):
        for b in batches:
            if b + batchSize < len(Y):
                x = X[perm[b : b + batchSize]]
                y = Y[perm[b : b + batchSize]]
            else:
                x = X[perm[b : ]]
                y = Y[perm[b : ]]
            loss = model.train_on_batch(x, y)

        print("\t\tEpoch %i. [Error, Accuracy]: %.15f, %.15f " % (e+1, loss[0], loss[1]))
        errLoss.append(loss[0])
        accLoss.append(loss[1])
   
    if not (os.path.exists("_FineTuning/")):
        os.makedirs("_FineTuning/")

    model.save_weights('_FineTuning/resnet_weights.h5')
    model.save('_FineTuning/resnet_model.h5')
    #print(model.metrics_names)

    print("\tPloting training loss ...")
    plt.plot(errLoss, label="Err")
    plt.plot(accLoss, label="Acc")
    plt.xlim([0, len(errLoss)-1])
    plt.legend(loc='upper right')
    plt.ylabel('Loss/Accuracy')
    plt.xlabel('Epochs')
    plt.margins(0.5, 0.5)
    plt.savefig("_FineTuning/resnet_training.png")
    plt.close()

def testingModel(X, Y, batchSize):
    print("\tTesting the model ...")

    model = load_model('_FineTuning/resnet_model.h5')
    model.load_weights('_FineTuning/resnet_weights.h5')
    acc = model.evaluate(X, Y, batch_size=batchSize)
    #print(model.metrics_names)
    print("\t\tTop-1 Accuracy: %f" % acc[1])

#--------------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    batchSize = 32
    numberEpochs = 10
    numberClasses = 10

    Xtrain, Ytrain, Xtest, Ytest = loadingImages()
    model = setModel(numberClasses)
    trainingModel(model, Xtrain, Ytrain, batchSize, numberEpochs)
    testingModel(Xtest, Ytest, batchSize)