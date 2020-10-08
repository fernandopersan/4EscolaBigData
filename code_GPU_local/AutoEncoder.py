import os
import numpy as np
from PIL import Image

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Reshape, UpSampling2D, Conv2DTranspose
from keras.models import load_model
from keras.datasets import cifar10

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------------------------------------------------------------------------------------------

def loadingImages():
    print("\tLoading CIFAR10 images ...")
    (Xtrain, Ytrain), (Xtest, Ytest) = cifar10.load_data()

    Xtrain, Ytrain = lowSampleDataset(Xtrain, Ytrain)
    Xtest, Ytest = lowSampleDataset(Xtest, Ytest)

    print('\t\tTraining set shape: ', Xtrain.shape)
    print('\t\tTesting set shape: ', Xtest.shape)
    return Xtrain/255, Ytrain, Xtest/255, Ytest

def lowSampleDataset(X, Y):
    perm = np.random.permutation(X.shape[0])
    X = X[perm[0 : (int)(X.shape[0] * (5/100))]]
    Y = Y[perm[0 : (int)(Y.shape[0] * (5/100))]]
    return X, Y

def definingAutoEncoder():
	print("\tDefining the AE ...")
	input_img = Input(shape=(32, 32, 3,))

	encoder = Conv2D(8, kernel_size=(3,3), strides=(2, 2), padding='same', activation='relu')(input_img)
	encoder = Conv2D(8, kernel_size=(3,3), padding='valid', activation='relu')(encoder)
	encoder = MaxPooling2D(pool_size=(2, 2))(encoder)
	encoder = Flatten(name='code')(encoder)

	decoder = Reshape((7,7,8))(encoder)
	decoder = UpSampling2D((2,2))(decoder)
	decoder = Conv2DTranspose(8, kernel_size=(3,3), padding='valid', activation='relu')(decoder)
	decoder = Conv2DTranspose(3, kernel_size=(3,3), strides=(2, 2), padding='same', activation='relu')(decoder)

	autoencoder = Model(input_img, decoder)
	autoencoder.compile(loss='mean_squared_error', optimizer='adam')
	autoencoder.summary()
	return autoencoder

def trainingAE(Xtrain, autoencoder, batchSize, numberEpochs):
	if not (os.path.exists("_FineTuning/")):
		os.makedirs("_FineTuning/")
	
	print("\tTraining the AE ...")
	historyAE = autoencoder.fit(x=Xtrain, y=Xtrain, batch_size=batchSize, epochs=numberEpochs, shuffle=True, verbose=1)
	autoencoder.save_weights('_FineTuning/ae_weights.h5')
	autoencoder.save('_FineTuning/ae_model.h5')

	print("\tPloting the training performance ...")
	plt.plot(historyAE.history['loss'])
	plt.ylabel('Loss')
	plt.legend(['AE'], loc='upper right')
	plt.savefig("_FineTuning/ae_training.png")
	plt.close()

def featureExtractionCNN(Xtrain, Xtest):
	print("\tFeature extraction with AutoEncoder ...")
	ae = load_model('_FineTuning/ae_model.h5')
	ae.load_weights('_FineTuning/ae_weights.h5')
	ae = Model(inputs=ae.input, outputs=ae.get_layer(name='code').output)

	prediction = np.array(ae.predict(Xtrain))
	Xtrain = np.reshape(prediction, (prediction.shape[0], prediction.shape[1]))

	prediction = np.array(ae.predict(Xtest))
	Xtest = np.reshape(prediction, (prediction.shape[0], prediction.shape[1]))

	print('\t\tFeatures training shape: ', Xtrain.shape)
	print('\t\tFeatures testing shape: ', Xtest.shape)
	return Xtrain, Xtest
	
def classificationSVM(Xtrain, Ytrain, Xtest, Ytest):
    print("\tClassification with Linear SVM ...")
    svm = SVC(kernel='linear')
    svm.fit(Xtrain, np.ravel(Ytrain, order='C'))
    result = svm.predict(Xtest)
    
    acc = accuracy_score(result, np.ravel(Ytest, order='C'))
    print("\t\tAccuracy Linear SVM: %0.4f" % acc)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
	batchSize = 32
	numberEpochs = 10

	Xtrain, Ytrain, Xtest, Ytest = loadingImages()
	autoencoder = definingAutoEncoder()
	trainingAE(Xtrain, autoencoder, batchSize, numberEpochs)
	Xtrain, Xtest = featureExtractionCNN(Xtrain, Xtest)
	classificationSVM(Xtrain, Ytrain, Xtest, Ytest)