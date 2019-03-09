####################################################
#### Unsupervised learning in MNIST
#### Cited: http://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/
####################################################
import pandas as pd
import numpy as np
from sklearn import cluster
import sklearn.metrics

from keras.datasets import mnist
(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX.reshape((trainX.shape[0], -1))/255
testX = testX.reshape((testX.shape[0],-1))/255

####################################################
#### 1. K-means
####################################################
for N in [6000,7000,8000,9000,10000,30000,60000]:
    kms = cluster.KMeans(n_clusters=10)
    kms.fit(trainX[0:N,])
    predTest = kms.predict(testX)
    a = acc(testY,predTest)
    print("N=%d" %N,":",a)

####################################################
#### 2. GMM
####################################################
from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold
from sklearn.mixture import GaussianMixture

N=2000
for x in ['spherical', 'diag', 'tied', 'full']:
    gmm=GaussianMixture(n_components=10, covariance_type=x)
    gmm.fit(trainX[0:N])
    predTest = gmm.predict(testX)
    a = acc(testY,predTest)
    print("%s" %x,"N=%d" %N,":",a)

for N in [6000,7000,8000,9000,10000,30000,60000]:
    gmm = GaussianMixture(n_components=10,covariance_type='spherical')
    gmm.fit(trainX[0:N,])
    predTest = gmm.predict(testX)
    a = acc(testY,predTest)
    print("N=%d" %N,":",a)


####################################################
#### 3. PCA-reduced data
####################################################
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
pca.fit(trainX[0:1000]) # I don't think pca is a good way, since variance are nearly evenly shared.
print(pca.explained_variance_ratio_)
print(pca.singular_values_)  

####################################################
#### Used to calculate accuracy
#### WARNING: different methods result in different accuracy
####################################################
# Cited: https://www.dlology.com/blog/how-to-do-unsupervised-clustering-with-keras/
def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    #y_true = y_true.as_matrix()
    #y_true = y_true.reshape((-1,)) 
    assert y_pred.shape == y_true.shape
    D = max(y_pred.max(), y_true.max()) + 1 # Dimension
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1 # contigency table
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w) # An algorithm
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

#### Confusion matrix
import seaborn as sns
import sklearn.metrics
import matplotlib.pyplot as plt
sns.set(font_scale=3)
confusion_matrix = sklearn.metrics.confusion_matrix(testY, predTest)

#### Method 3 adjusted Rand index ### To be discussed
