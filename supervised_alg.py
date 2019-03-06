# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 13:58:53 2019

@author: Xueyu Shi
"""

import numpy as np
import os
from sklearn import neighbors, svm, linear_model, tree, ensemble, naive_bayes, neural_network
from sklearn import model_selection


def Knn_classify(dataset):
    n_neighbors = np.array([5, 10, 15, 20])
    
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
    regular_para = np.array([1, 10, 100])
    
    for C in regular_para:
        unlabel_error=[]
        test_error=[]
        models = (svm.SVC(kernel='linear', C=C),
                  svm.LinearSVC(C=C),
                  svm.SVC(kernel='rbf', gamma=10, C=C),
                  svm.SVC(kernel='poly', degree=3, C=C))
        models = (clf.fit(dataset['train_labled_x'], dataset['train_labled_y']) for clf in models)
        
        for clf in models:
            unlabel_error.append(clf.score(dataset['train_unlabled_x'], dataset['train_unlabled_y']))
            test_error.append(clf.score(dataset['test_x'], dataset['test_y']))
        
        print(C, unlabel_error)
        print(C, test_error)
            
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
              neural_network.MLPClassifier(hidden_layer_sizes=(1024, 512), max_iter=500))
    
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


def main():
    total_data = 2000
    normalize_type = 'pca500'
    file_dir = 'small_dataset'
    test_file = os.path.join(file_dir, 'cifar10-test-all-' + normalize_type +'.csv')
    ps = np.array([10, 30, 50])
    
    data_test = np.genfromtxt(test_file, dtype=np.float64, delimiter=',')
    print('test')
    
    for p in ps:
        label_file = os.path.join(file_dir, 'cifar10-total' + str(total_data) +'-label'+str(p) + '-' + normalize_type+'-labeled.csv')
        unlabel_file = os.path.join(file_dir, 'cifar10-total' + str(total_data) +'-label'+str(p) + '-' + normalize_type+'-unlabeled.csv')
        dataset = load_data(label_file, unlabel_file, test_file)
        
        print('train_load')
#        print(dataset['train_labled_x'].shape, dataset['train_labled_y'].shape)
#        print(dataset['train_unlabled_x'].shape, dataset['train_unlabled_y'].shape)
#        print(dataset['test_x'].shape, dataset['test_y'].shape)
        dataset['test_x'] = data_test[:, 1:].astype(np.float64)
        dataset['test_y'] = data_test[:, 0].astype(np.float64)
        print('test_1')
        
        print(p, 'KNN Results:')
        Knn_classify(dataset)
#        
        print(p, 'SVM Results:')
        SVM_classify(dataset)
        
        print(p, 'LinearModel Results:')
        
        LinearModel_classify(dataset)
        
#        print(p, 'DecisionTree Results:')
#        DecisionTrees_classify(dataset)
        
        print(p, 'DNN Results:')
        NeuralNet_classify(dataset)

        
        
    
    
    
    
    
    
if __name__ == '__main__':
    main()