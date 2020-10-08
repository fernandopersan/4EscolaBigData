import argparse
import numpy as np
from PIL import Image

from keras.datasets import cifar10
from keras.models import Model
from keras.applications import resnet50

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

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

    print('\t\tTraining set shape: ', Xtrain.shape)
    print('\t\tTesting set shape: ', Xtest.shape)
    return Xtrain, Ytrain, Xtest, Ytest

def lowSampleDataset(X, Y):
    perm = np.random.permutation(X.shape[0])
    X = X[perm[0 : (int)(X.shape[0] * (5/100))]]
    Y = Y[perm[0 : (int)(Y.shape[0] * (5/100))]]
    return X, Y

def featureExtractionCNN(Xtrain, Xtest):
    print("\tLoading the ResNet50-ImageNet model ...")
    model = resnet50.ResNet50(include_top=True, weights='imagenet', input_shape=(224, 224, 3), classes=1000)
    model = Model(inputs=model.input, outputs=model.get_layer(index=-2).output)
    #model.summary()
    
    prediction = np.array(model.predict(Xtrain))
    Xtrain = np.reshape(prediction, (prediction.shape[0], prediction.shape[1]))
    
    prediction = np.array(model.predict(Xtest))
    Xtest = np.reshape(prediction, (prediction.shape[0], prediction.shape[1]))
    
    print('\t\tFeatures training shape: ', Xtrain.shape)
    print('\t\tFeatures testing shape: ', Xtest.shape)
    return Xtrain, Xtest

def crossValidation(Xtrain, Ytrain):
    print("\tCross-validation with K-NN ...")
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    #kf = KFold(n_splits=5, shuffle=True)

    knn = KNeighborsClassifier(n_neighbors=3)
    scores = cross_val_score(knn, Xtrain, np.ravel(Ytrain, order='C'), cv=kf)
    print('\t\tAccuracy K-NN: %0.4f +/- %0.4f' % (scores.mean(), scores.std()))

def dimensionReduction(Xtrain, Xtest):
    print("\tDimensionality reduction with PCA ...")
    pca = PCA(n_components=256)
    Xtrain = pca.fit_transform(Xtrain)
    Xtest = pca.transform(Xtest)

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

def multiFeatureExtractionCNN(Xtrain, Xtest):
    print("\tLoading the ResNet50-ImageNet model ...")
    model = resnet50.ResNet50(include_top=True, weights='imagenet', input_shape=(224, 224, 3), classes=1000)

    modelGlobal = Model(inputs=model.input, outputs=model.get_layer(name='avg_pool').output)
    modelLocal = Model(inputs=model.input, outputs=model.get_layer(name='activation_4').output)
    
    prediction = np.array(modelGlobal.predict(Xtrain))
    XtrainGlobal = np.reshape(prediction, (prediction.shape[0], prediction.shape[1]*prediction.shape[2]*prediction.shape[3]))
    
    prediction = np.array(modelGlobal.predict(Xtest))
    XtestGlobal = np.reshape(prediction, (prediction.shape[0], prediction.shape[1]*prediction.shape[2]*prediction.shape[3]))

    prediction = np.array(modelLocal.predict(Xtrain))
    XtrainLocal = np.reshape(prediction, (prediction.shape[0], prediction.shape[1]*prediction.shape[2]*prediction.shape[3]))
    
    prediction = np.array(modelLocal.predict(Xtest))
    XtestLocal = np.reshape(prediction, (prediction.shape[0], prediction.shape[1]*prediction.shape[2]*prediction.shape[3]))
    
    Xtrain = np.concatenate((XtrainGlobal, XtrainLocal), axis=1)
    Xtest = np.concatenate((XtestGlobal, XtestLocal), axis=1)

    print('\t\tFeatures fusion training shape: ', Xtrain.shape)
    print('\t\tFeatures fusion testing shape: ', Xtest.shape)
    return Xtrain, Xtest
    
#--------------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scenario', help="Insert the scenario to execute", type=int)
    args = parser.parse_args()

    if (args.scenario == 1):
        Xtrain, Ytrain, Xtest, Ytest = loadingImages()
        Xtrain, Xtest = featureExtractionCNN(Xtrain, Xtest)
        #classificationSVM(Xtrain, Ytrain, Xtest, Ytest)
        crossValidation(Xtrain, Ytrain)
    
    elif (args.scenario == 2):
        Xtrain, Ytrain, Xtest, Ytest = loadingImages()
        Xtrain, Xtest = featureExtractionCNN(Xtrain, Xtest)
        Xtrain, Xtest = dimensionReduction(Xtrain, Xtest)
        classificationSVM(Xtrain, Ytrain, Xtest, Ytest)

    elif (args.scenario == 3):
        Xtrain, Ytrain, Xtest, Ytest = loadingImages()
        Xtrain, Xtest = multiFeatureExtractionCNN(Xtrain, Xtest)
        Xtrain, Xtest = dimensionReduction(Xtrain, Xtest)
        classificationSVM(Xtrain, Ytrain, Xtest, Ytest)