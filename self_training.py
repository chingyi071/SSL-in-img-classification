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
        self.min_threshold   = 300

    def __str__(self):
        return "Classifier: " + self.name + "\nParameters: " + str(self.base_classifier.get_params())

    def fit(self, dataset):
        Xtrain = np.vstack((dataset['train_labeled_x'], dataset['train_unlabeled_x']))
        ytrain = np.hstack((dataset['train_labeled_y'], ['unlabeled']*len(dataset['train_unlabeled_x'])))
        Xtest = dataset['test_x']
        ytest = dataset['test_y']
        X = np.copy(Xtrain)
        y = np.copy(ytrain) # copy in order not to change original data
        
        all_are_labeled = False
        iteration = 0
        train_labeled_losses, train_unlabeled_losses, test_losses = [], [], []
        train_labeled_accuraciess, train_unlabeled_accuraciess, test_accuraciess = [], [], []
        # Iterate until the result is stable or max_iterations is reached
        while not all_are_labeled and (iteration < self.max_iterations):
            self._fit_iteration(X, y)
            all_are_labeled = (y != "unlabeled").all()
            iteration += 1
            train_labeled_loss, train_labeled_error, train_labeled_accuracy = \
                                self.score(dataset['train_labeled_x'], dataset['train_labeled_y'])
            train_unlabeled_loss, train_unlabeled_error, train_unlabeled_accuracy = \
                                self.score(dataset['train_unlabeled_x'], dataset['train_unlabeled_y'])
            test_loss, test_error, test_accuracy = self.score(dataset['test_x'], dataset['test_y'])
            
            # Record the loss
            train_labeled_losses.       append(train_labeled_loss)
            train_unlabeled_losses.     append(train_unlabeled_loss)
            test_losses.                append(test_loss)
            train_labeled_accuraciess.  append(train_labeled_accuracy) 
            train_unlabeled_accuraciess.append(train_unlabeled_accuracy) 
            test_accuraciess.           append(test_accuracy)

            print('iteration = '+str(iteration)+', train_unlabeled_accuracy = '+str(train_unlabeled_accuracy)
                  +', test_accuracy = '+str(test_accuracy))
        return train_labeled_losses, train_unlabeled_losses, test_losses, \
               train_labeled_accuraciess, train_unlabeled_accuraciess, test_accuraciess
        
    def gmm_analysis(self, X, y, labeled):
        from sklearn.mixture import GaussianMixture
        threshold = 0.7

        # New-labeled data selection
        # Calculate the clustering score
        gmm = GaussianMixture(n_components=10, covariance_type='spherical')
        gmm.fit(X[labeled], y[labeled])
        clstr_score = gmm.score_samples(X)
        print("prob varies from %f to %f" % (min(clstr_score), max(clstr_score)))
        
        # Filter the result over threshold
        threshold = min( self.min_threshold, clstr_score[~labeled].max())
        print("Number of unlabled data = ", clstr_score[~labeled].shape, "/", clstr_score.shape)
        predicted_index = gmm.predict(X)
        return predicted_index, clstr_score

    def label_assign( self, X, y, gmm_predicted ):
        lut = [None]*10 # Indexed by cluster number get real label
        print(    "                True label   ", "   ".join(["%d"%x for x in range(10)]))
        for i in range(10): # i = label
            data_labeled_with_label_i = (y == str(i))
            predicted = gmm_predicted[data_labeled_with_label_i]
            count = [len([x for x in predicted if x==clstr_index]) for clstr_index in range(10)]
            print("Distribution with %d-label = " % i, ", ".join(["%2d" % x for x in count]), ", the max is ", count.index(max(count)))
            label_with_max_count = count.index(max(count))
            lut[label_with_max_count] = str(i)
        print("lut = ", ", ".join(["%d->'%s'" % (cluster_index, label) for cluster_index, label in enumerate(lut)]))
        return lut

    def _fit_iteration(self, X, y):
        from sklearn.mixture import GaussianMixture
        threshold = 0.7
        labeled = (y != 'unlabeled')

        gmm_predicted, clstr_score = self.gmm_analysis(X,y,labeled)

        clf = self.base_classifier
        labeled = (y != 'unlabeled')
        clf.fit( X[labeled], y[labeled] )

        lut = self.label_assign(X,y,gmm_predicted)

        import statistics
        for i in range(10):
            if lut[i] is not None:
                with_gmm_label_i = gmm_predicted==i
                # print("Number in cluster %d" % i, y[with_gmm_label_i].shape, "ranged from %d to %d" % (min(clstr_score[with_gmm_label_i]), max(clstr_score[with_gmm_label_i])))
                stdev = statistics.stdev(clstr_score[with_gmm_label_i])
                mean = statistics.mean(clstr_score[with_gmm_label_i])
                over_per_cluster_thresh = clstr_score >= (mean+2*stdev)
                if y[~labeled & over_per_cluster_thresh & with_gmm_label_i].shape[0] > 0:
                    y[~labeled & over_per_cluster_thresh & with_gmm_label_i] = lut[i]
                    print("Label", y[~labeled & over_per_cluster_thresh & with_gmm_label_i].shape, "in cluster %d to" % i, "'%s'" % lut[i]\
                        , "with real = ", y[~labeled & over_per_cluster_thresh & with_gmm_label_i])

        labeled = (y != 'unlabeled')
        print("labeled size = ", len([x for x in labeled if x == True]))

#        y[~labeled & over_thresh] = gmm.predict(X[~labeled & over_thresh])

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

    dataset_path = '/home/chingyi/Datasets/mnist'
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
    
    Train_labeled_loss,     Train_unlabeled_loss,     Test_loss     = {}, {}, {}
    Train_labeled_accuracy, Train_unlabeled_accuracy, Test_accuracy = {}, {}, {}
    if os.path.isfile(result_file+'.pickle'):
        with open(result_file+'.pickle', 'rb') as file:
            results = pickle.load(file)
            Train_labeled_loss       = results['Train_labeled_loss']
            Train_unlabeled_loss     = results['Train_unlabeled_loss']
            Test_loss                = results['Test_loss']
            Train_labeled_accuracy   = results['Train_labeled_accuracy']
            Train_unlabeled_accuracy = results['Train_unlabeled_accuracy']
            Test_accuracy            = results['Test_accuracy']

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        if name in Train_labeled_loss:
            continue
        else:
            Train_labeled_loss[name], Train_unlabeled_loss[name], Test_loss[name] = [], [], []
            Train_labeled_accuracy[name], Train_unlabeled_accuracy[name], Test_accuracy[name] = [], [], []

        for size in sizes:
            num_label_data = int( n_samples * size / 100 )
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
