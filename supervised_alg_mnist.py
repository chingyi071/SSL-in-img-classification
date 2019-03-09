# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 21:50:36 2019

@author: Xueyu Shi
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 13:58:53 2019

@author: Xueyu Shi
"""

import numpy as np
import os
from sklearn import neighbors, svm, linear_model, tree, ensemble, naive_bayes, neural_network
from sklearn import model_selection
import timeit


def Supervised_models(dataset):
    
    svc_rbf = svm.SVC(kernel='rbf')
    rbf_params = {"C":[1, 5, 10], "gamma": [0.01, 0.001]}
    rbf_clf = model_selection.GridSearchCV(svc_rbf, rbf_params)
    
    models = (linear_model.LogisticRegression(random_state=0, solver='sag', multi_class='multinomial', max_iter=500),
              neighbors.KNeighborsClassifier(5, weights='uniform'),
              svm.SVC(kernel='linear', C=0.1),
              rbf_clf,
              neural_network.MLPClassifier(hidden_layer_sizes=(512, 512,512), max_iter=500),
              neural_network.MLPClassifier(hidden_layer_sizes=(512, 512,512, 216), max_iter=1000))
    models = (clf.fit(dataset['train_labled_x'], dataset['train_labled_y']) for clf in models)
    
    label_acc = []
    unlabel_acc = []
    test_acc = []
    
    for clf in models:
        label_acc.append(clf.score(dataset['train_labled_x'], dataset['train_labled_y']))
        unlabel_acc.append(clf.score(dataset['train_unlabled_x'], dataset['train_unlabled_y']))
        test_acc.append(clf.score(dataset['test_x'], dataset['test_y']))
    
    print('Labeled accuracy: ', label_acc)
    print('Unlabeled accuracy: ', unlabel_acc)
    print('Test accuracy: ', test_acc)
        

def Knn_classify(dataset):
    n_neighbors = np.array([5, 10, 20, 50])
    
    unlabel_error=[]
    test_error=[]
    
    for n in n_neighbors:
        clf = neighbors.KNeighborsClassifier(n, weights='uniform')
        clf.fit(dataset['train_labled_x'], dataset['train_labled_y'])

        unlabel_error.append(clf.score(dataset['train_unlabled_x'], dataset['train_unlabled_y']))
        test_error.append(clf.score(dataset['test_x'], dataset['test_y']))
    
    print(unlabel_error)
    print(test_error)

def SVM_classify(dataset):
    regular_para = np.array([10, 100])
    gammas = np.array([10, 1, 0.1, 0.01, 0.001])
    C = 10
    
#    for C in regular_para:
    for gamma_para in gammas:
        unlabel_error=[]
        test_error=[]
        models = (#svm.SVC(kernel='linear', C=C),
                  #svm.LinearSVC(C=C),
                  svm.SVC(kernel='rbf', gamma=gamma_para, C=C),
                  svm.SVC(kernel='rbf'),
                  svm.SVC(kernel='poly', degree=3, C=C))
        models = (clf.fit(dataset['train_labled_x'], dataset['train_labled_y']) for clf in models)
        
        for clf in models:
            unlabel_error.append(clf.score(dataset['train_unlabled_x'], dataset['train_unlabled_y']))
            test_error.append(clf.score(dataset['test_x'], dataset['test_y']))
        
        print(C, unlabel_error)
        print(C, test_error)
    
    # svm RBF kernel
#    svc = svm.SVC(kernel='rbf')
#    params = {"C":[1, 10, 100], "gamma": [10, 1, 0.1, 0.01, 0.001]}
#    clf = model_selection.GridSearchCV(svc, params)
#    
#    clf.fit(dataset['train_labled_x'], dataset['train_labled_y'])
#    print(clf.get_params())
#    
##    print(clf.get_params())
#    print(clf.score(dataset['train_unlabled_x'], dataset['train_unlabled_y']), clf.score(dataset['test_x'], dataset['test_y']))
#            
def LinearModel_classify(dataset):
    models = (linear_model.LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial'),
              linear_model.Perceptron(tol=1e-3, random_state=0),
              ensemble.RandomForestClassifier(n_estimators=10, max_depth=50, random_state=0),
              naive_bayes.GaussianNB())
    models = (clf.fit(dataset['train_labled_x'], dataset['train_labled_y']) for clf in models)
    

    test_acc = []
    unlabel_error = []
    for clf in models:
        test_acc.append(clf.score(dataset['test_x'], dataset['test_y']))
        unlabel_error.append(clf.score(dataset['train_unlabled_x'], dataset['train_unlabled_y']))
    
    print(unlabel_error)
    print(test_acc)
     
    
def DecisionTrees_classify(dataset):
    clf = tree.DecisionTreeClassifier(random_state=0)
    model_selection.cross_val_score(clf, dataset['train_labled_x'], dataset['train_labled_y'])
    
    print clf.score(dataset['test_x'], dataset['test_y'])
    


def NeuralNet_classify(dataset):
    models = (neural_network.MLPClassifier(hidden_layer_sizes=(512, 216), max_iter=500),
              neural_network.MLPClassifier(hidden_layer_sizes=(512, 216, 216), max_iter=500),
              neural_network.MLPClassifier(hidden_layer_sizes=(512, 512,512), max_iter=500))
    
    models = (clf.fit(dataset['train_labled_x'], dataset['train_labled_y']) for clf in models)
    
    unlabel_acc = []
    test_acc = []
    for clf in models:
        unlabel_acc.append(clf.score(dataset['train_unlabled_x'], dataset['train_unlabled_y']))
        test_acc.append(clf.score(dataset['test_x'], dataset['test_y']))
#        print unlabel_acc
    
    print (unlabel_acc, test_acc)



def load_data(label_file, unlabel_file, test_file):
    data_labeled = np.genfromtxt(label_file, dtype=np.float64, delimiter=',')
    data_unlabeled = np.genfromtxt(unlabel_file, dtype=np.float64, delimiter=',')
#    data_test = np.genfromtxt(test_file, dtype=np.float64, delimiter=',')
    
    return {
        'train_labled_x': data_labeled[:, 1:].astype(np.float64),
        'train_labled_y': data_labeled[:, 0].astype(np.int64),
        'train_unlabled_x': data_unlabeled[:, 1:].astype(np.float64),
        'train_unlabled_y': data_unlabeled[:, 0].astype(np.int64),
#        'test_x': data_test[:, 1:].astype(np.float64),
#        'test_y': data_test[:, 0].astype(np.int64),
    }

#def NaiveBayes_classify(dataset):
#    models = (naive_bayes.GaussianNB(),
#              naive_bayes.MultinomialNB(),
#              ComplementNB())
#    models = (clf.fit(dataset['train_labled_x'], dataset['train_labled_y']) for clf in models)
#    test_acc = []
#    unlabel_error = []
#    for clf in models:
##       yhat_unlabel = clf.predict(dataset['train_unlabled_x'])
##       unlabel_error.append(float((yhat_unlabel == dataset['train_unlabled_y']).mean()))
#        test_acc.append(clf.score(dataset['test_x'], dataset['test_y']))
#    
#    print test_acc

import gzip
def data_prepare(dataset_path, n_samples):
    image_size = 28

    num_images = 60000
    x_train_file = gzip.open(dataset_path+'/train-images-idx3-ubyte.gz','r')
    x_train_file.read(16)
    x_train_buf = x_train_file.read(image_size * image_size * num_images)
    x_train_all = np.frombuffer(x_train_buf, dtype=np.uint8).reshape(num_images, image_size * image_size) / 255.0
    

    y_train_file = gzip.open(dataset_path+'/train-labels-idx1-ubyte.gz','r')
    y_train_file.read(8)
    y_train_buf = y_train_file.read(num_images * 1)
#    print(np.frombuffer(y_train_buf, dtype=np.uint8).shape)
    y_train_all = np.frombuffer(y_train_buf, dtype=np.uint8).reshape(num_images, )
    
    num_images = 10000
    x_test_file = gzip.open(dataset_path+'/t10k-images-idx3-ubyte.gz','r')
    x_test_file.read(16)
    x_test_buf = x_test_file.read(image_size * image_size * num_images)
    x_test_all = np.frombuffer(x_test_buf, dtype=np.uint8).reshape(num_images, image_size * image_size) / 255.0

    y_test_file = gzip.open(dataset_path+'/t10k-labels-idx1-ubyte.gz','r')
    y_test_file.read(8)
    y_test_buf = y_test_file.read(num_images * 1)
#    print(np.frombuffer(y_test_buf, dtype=np.uint8).shape)
    y_test_all = np.frombuffer(y_test_buf, dtype=np.uint8).reshape(num_images, )
    return {
        'train_x': x_train_all.astype(np.float64),
        'train_y': y_train_all,
        'test_x': x_test_all.astype(np.float64),
        'test_y': y_test_all,
    }


def main():
    
    dataset_path = 'Datasets/mnist'
    
    n_samples = 60000
    
    dataset_all = data_prepare(dataset_path, n_samples)
    
#    ps = np.array([10, 30, 50])
    ps = np.array([50])
    
    print(dataset_all['train_x'].shape)
    dataset = {}
#    print(dataset_all['train_x'][0,])
    for p in ps:

        num_label_data = n_samples * p / 100
        dataset['train_labled_x'] = dataset_all['train_x'][:num_label_data,]
        dataset['train_labled_y'] = dataset_all['train_y'][:num_label_data]
        dataset['train_unlabled_x'] = dataset_all['train_x'][num_label_data:n_samples,]
        dataset['train_unlabled_y'] = dataset_all['train_y'][num_label_data:n_samples]
        
        dataset['test_x'] = dataset_all['test_x']
        dataset['test_y'] = dataset_all['test_y']
        
#        print(dataset['train_labled_x'].shape, dataset['train_labled_y'].shape)
#        print(dataset['train_unlabled_x'].shape, dataset['train_unlabled_y'].shape)
#        print(dataset['test_x'].shape, dataset['test_y'].shape)
        
#        print('test_1')
        
#        print(p, 'KNN Results:')
#        start = timeit.timeit()
#        Knn_classify(dataset)
#        end = timeit.timeit()
#        print (p, 'KNN time:', end - start)
        
#        
#        print(p, 'SVM Results:')
#        start = timeit.timeit()
#        SVM_classify(dataset)
#        end = timeit.timeit()
#        print (p, 'SVM time:', end - start)
        
#        print(p, 'LinearModel Results:')
#        start = timeit.timeit()
#        LinearModel_classify(dataset)
#        end = timeit.timeit()
#        print (p, 'LinearModel time:', end - start)
#        
#        
#        
##        print(p, 'DecisionTree Results:')
##        DecisionTrees_classify(dataset) 
#        
#        print(p, 'DNN Results:')
#        start = timeit.timeit()
#        NeuralNet_classify(dataset)
#        end = timeit.timeit()
#        print (p, 'DNN time:', end - start)
        
        print(p, 'Supervised Results:')
        Supervised_models(dataset)
        

        
        
    
    
    
    
    
    
if __name__ == '__main__':
    main()