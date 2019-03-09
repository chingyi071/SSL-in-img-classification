import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import timeit
import pickle
from pickle import dump
from matplotlib.colors import ListedColormap
from sklearn import neighbors, svm, linear_model, tree, ensemble, naive_bayes, neural_network
from sklearn import model_selection



class StandardSelfTraining:
    def __init__(self, name, base_classifier, max_iterations=10):
        self.name = name
        self.base_classifier = base_classifier
        self.max_iterations = max_iterations

    def __str__(self):
        return "Classifier: " + self.name + "\nParameters: " + str(self.base_classifier.get_params())

    def fit(self, dataset):
        Xtrain = np.vstack((dataset['train_labeled_x'], dataset['train_unlabeled_x']))
        ytrain = np.hstack((dataset['train_labeled_y'], ['unlabeled']*len(dataset['train_unlabeled_x'])))
        Xtest = dataset['test_x']
        ytest = dataset['test_y']
        X = np.copy(Xtrain)
        y = np.copy(ytrain) # copy in order not to change original data
        
        all_labeled = False
        iteration = 0
        train_labeled_losses, train_unlabeled_losses, test_losses = [], [], []
        train_labeled_accuraciess, train_unlabeled_accuraciess, test_accuraciess = [], [], []
        # Iterate until the result is stable or max_iterations is reached
        while not all_labeled and (iteration < self.max_iterations):
            self._fit_iteration(X, y)
            all_labeled = (y != "unlabeled").all()
            iteration += 1
            train_labeled_loss, train_labeled_error, train_labeled_accuracy = \
                                self.score(dataset['train_labeled_x'], dataset['train_labeled_y'])
            train_unlabeled_loss, train_unlabeled_error, train_unlabeled_accuracy = \
                                self.score(dataset['train_unlabeled_x'], dataset['train_unlabeled_y'])
            test_loss, test_error, test_accuracy = self.score(dataset['test_x'], dataset['test_y'])
            train_labeled_losses.append(train_labeled_loss)
            train_unlabeled_losses.append(train_unlabeled_loss)
            test_losses.append(test_loss)
            train_labeled_accuraciess.append(train_labeled_accuracy) 
            train_unlabeled_accuraciess.append(train_unlabeled_accuracy) 
            test_accuraciess.append(test_accuracy)
            print('iteration = '+str(iteration)+', train_unlabeled_accuracy = '+str(train_unlabeled_accuracy)
                  +', test_accuracy = '+str(test_accuracy))
        return train_labeled_losses, train_unlabeled_losses, test_losses, \
               train_labeled_accuraciess, train_unlabeled_accuraciess, test_accuraciess
        
    def _fit_iteration(self, X, y):
        threshold = 0.7
        
        clf = self.base_classifier
        # Fit a classifier on already labeled data
        labeled = (y != 'unlabeled')
        clf.fit(X[labeled], y[labeled])

        probabilities = clf.predict_proba(X)
        threshold = min(threshold, probabilities[~labeled, :].max()) # Get at least the best one
        over_thresh = probabilities.max(axis=1)>=threshold
        
        y[~labeled & over_thresh] = clf.predict(X[~labeled & over_thresh])

    def predict(self, X):
        yhat = self.base_classifier.predict(X)
        return yhat

    def score(self, X, y):
        clf = self.base_classifier
        probabilities = clf.predict_proba(X)
        ytrue = np.zeros_like(probabilities)
        for i in range(len(y)):
            ytrue[i, int(y[i])] = 1
        loss = - np.multiply(ytrue, np.log(probabilities)).sum() / float(len(y))
        yhat = self.base_classifier.predict(X)
        error = np.sum(np.not_equal(yhat.astype(int), y.astype(int))) / float(len(y))
        accuracy = 1.-error
        # print(np.hstack((y.reshape((-1,1)), yhat.reshape((-1,1)))))
        return loss, error, accuracy



# ================================================================================== #
#                              Data Prepare and Loading                              #
# ================================================================================== #

def load_data(label_file, unlabel_file, test_file):
    data_labeled = np.genfromtxt(label_file, dtype=np.float64, delimiter=',')
    data_unlabeled = np.genfromtxt(unlabel_file, dtype=np.float64, delimiter=',')
    return {'train_labeled_x': data_labeled[:, 1:].astype(np.float64),
            'train_labeled_y': data_labeled[:, 0].astype(np.int64),
            'train_unlabeled_x': data_unlabeled[:, 1:].astype(np.float64),
            'train_unlabeled_y': data_unlabeled[:, 0].astype(np.int64)}


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
    y_train_all = np.frombuffer(y_train_buf, dtype=np.uint8).reshape(num_images, )
    
    num_images = 10000
    x_test_file = gzip.open(dataset_path+'/t10k-images-idx3-ubyte.gz','r')
    x_test_file.read(16)
    x_test_buf = x_test_file.read(image_size * image_size * num_images)
    x_test_all = np.frombuffer(x_test_buf, dtype=np.uint8).reshape(num_images, image_size * image_size) / 255.0

    y_test_file = gzip.open(dataset_path+'/t10k-labels-idx1-ubyte.gz','r')
    y_test_file.read(8)
    y_test_buf = y_test_file.read(num_images * 1)
    y_test_all = np.frombuffer(y_test_buf, dtype=np.uint8).reshape(num_images, )
    return {'train_x': x_train_all,
            'train_y': y_train_all,
            'test_x': x_test_all,
            'test_y': y_test_all}



# ================================================================================== #
#                                       main                                         #
# ================================================================================== #
if __name__ == '__main__':

    dataset_path = '../data/mnist'
    n_samples = 6000
    max_iterations = 10
    result_file = 'self_training_mnist_normalized'+str(n_samples)

    dataset_all = data_prepare(dataset_path, n_samples)
    sizes = [10, 30, 50]
    print(dataset_all['train_x'].shape)
    dataset = {}
    dataset['test_x'] = dataset_all['test_x']
    dataset['test_y'] = dataset_all['test_y']
    
    names = ['DNN', 'SVM_rbf', 'SVM_linear', 'KNN5', 'logistics']
    classifiers = [neural_network.MLPClassifier(hidden_layer_sizes=(512, 216), max_iter=500),
                   svm.SVC(kernel='rbf', C=5, gamma=0.01, probability=True), 
                   svm.SVC(kernel='linear', C=0.1, probability=True),
                   neighbors.KNeighborsClassifier(5, weights='uniform'),
                   linear_model.LogisticRegression(random_state=0, solver='sag', multi_class='multinomial', max_iter=500),
                   # SVC(kernel="linear", C=0.025, probability=True),
                   # SVC(gamma=2, C=1, probability=True),
                   # DecisionTreeClassifier(max_depth=5),
                   # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                   # AdaBoostClassifier()
                   ]
    
    Train_labeled_loss, Train_unlabeled_loss, Test_loss = {}, {}, {}
    Train_labeled_accuracy, Train_unlabeled_accuracy, Test_accuracy = {}, {}, {}
    if os.path.isfile(result_file+'.pickle'):
        with open(result_file+'.pickle', 'rb') as file: results = pickle.load(file)
        Train_labeled_loss = results['Train_labeled_loss']
        Train_unlabeled_loss = results['Train_unlabeled_loss']
        Test_loss = results['Test_loss']
        Train_labeled_accuracy = results['Train_labeled_accuracy']
        Train_unlabeled_accuracy = results['Train_unlabeled_accuracy']
        Test_accuracy = results['Test_accuracy']

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        if name in Train_labeled_loss:
            continue
        else:
            Train_labeled_loss[name], Train_unlabeled_loss[name], Test_loss[name] = [], [], []
            Train_labeled_accuracy[name], Train_unlabeled_accuracy[name], Test_accuracy[name] = [], [], []

        for size in sizes:
            num_label_data = n_samples * size / 100
            dataset['train_labeled_x'] = dataset_all['train_x'][:num_label_data,]
            dataset['train_labeled_y'] = dataset_all['train_y'][:num_label_data]
            dataset['train_unlabeled_x'] = dataset_all['train_x'][num_label_data:n_samples,]
            dataset['train_unlabeled_y'] = dataset_all['train_y'][num_label_data:n_samples]

            method = StandardSelfTraining(name, clf, max_iterations=max_iterations)
            print(method.__str__())
            train_labeled_losses, train_unlabeled_losses, test_losses, \
               train_labeled_accuraciess, train_unlabeled_accuraciess, test_accuraciess = method.fit(dataset)

            Train_labeled_loss[name].append(train_labeled_losses)
            Train_unlabeled_loss[name].append(train_unlabeled_losses)
            Test_loss[name].append(test_losses)
            Train_labeled_accuracy[name].append(train_labeled_accuraciess)
            Train_unlabeled_accuracy[name].append(train_unlabeled_accuraciess)
            Test_accuracy[name].append(test_accuraciess)

        result_to_file = {'Train_labeled_loss': Train_labeled_loss,
                          'Train_unlabeled_loss': Train_unlabeled_loss,
                          'Test_loss': Test_loss,
                          'Train_labeled_accuracy': Train_labeled_accuracy,
                          'Train_unlabeled_accuracy': Train_unlabeled_accuracy,
                          'Test_accuracy': Test_accuracy}
        with open(result_file+'.txt', "w") as file: file.write(str(result_to_file))
        with open(result_file+'.pickle', "wb") as file: dump(result_to_file, file)