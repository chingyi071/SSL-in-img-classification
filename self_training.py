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
    def __init__(self, name, base_classifier, base_cluster, max_iterations=10):
        self.name = name
        self.base_classifier = base_classifier
        self.base_cluster    = base_cluster
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
        

    def _fit_iteration(self, X, y):
        from sklearn.mixture import GaussianMixture
        threshold = 0.7
        labeled = (y != 'unlabeled')


        clf = self.base_classifier
        labeled = (y != 'unlabeled')
        clf.fit( X[labeled], y[labeled] )

        label_index, label_value = self.base_cluster.selected_labeled(X,y,labeled)

        for i, non_label in enumerate(label_index):
            assert(non_label == False or y[i] == 'unlabeled')

        y[label_index] = label_value
        labeled = (y != 'unlabeled')
        print("labeled size = ", len([x for x in labeled if x == True]))

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

class GMM:
    def gmm_analysis( self, X, y, labeled):
        from sklearn.mixture import GaussianMixture
        threshold = 0.7

        # New-labeled data selection
        # Calculate the clustering score
        gmm = GaussianMixture(n_components=10, covariance_type='spherical')
        gmm.fit(X[labeled], y[labeled])
        clstr_score = gmm.score_samples(X)
        print("prob varies from %f to %f" % (min(clstr_score), max(clstr_score)))
        
        # Filter the result over threshold
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

    def selected_labeled( self, X, y, labeled):
        gmm_predicted, clstr_score = self.gmm_analysis(X,y,labeled)
        lut = self.label_assign(X,y,gmm_predicted)

        label_index = np.zeros_like(y, dtype=np.bool_)
        label_value = []
        import statistics
        for i in range(10):
            if lut[i] is not None:
                with_gmm_label_i = gmm_predicted==i
                # print("Number in cluster %d" % i, y[with_gmm_label_i].shape, "ranged from %d to %d" % (min(clstr_score[with_gmm_label_i]), max(clstr_score[with_gmm_label_i])))
                stdev = statistics.stdev(clstr_score[with_gmm_label_i])
                mean  = statistics.mean(clstr_score[with_gmm_label_i])
                over_per_cluster_thresh = clstr_score >= (mean+2*stdev)

                num_of_new_label = sum(~labeled & over_per_cluster_thresh & with_gmm_label_i)
                if  num_of_new_label > 0:
                    label_index[~labeled & over_per_cluster_thresh & with_gmm_label_i] = True
                    label_value.extend([lut[i]]*num_of_new_label)
                    print("Label", y[~labeled & over_per_cluster_thresh & with_gmm_label_i].shape, "in cluster %d to" % i, "'%s'" % lut[i]\
                        , "with real = ", y[~labeled & over_per_cluster_thresh & with_gmm_label_i])
        return label_index, np.array(label_value)

class DNN_cluster:
    def selected_labeled(self, X, y, labeled):

        # Make sure @labeled is correct
        for i, is_label in enumerate(labeled):
            assert(is_label == True or y[i] == 'unlabeled')

        # Define a DNN classfier
        dnn = neural_network.MLPClassifier(hidden_layer_sizes=(512, 216), max_iter=500)

        # Fit to data and get the prob for unlabeled data
        dnn.fit(X[labeled],y[labeled])
        probabilities = dnn.predict_proba(X)

        # Select the unlabeled data with prob greater than a threshold
        threshold = min(0.7, probabilities[~labeled, :].max())
        over_thresh = probabilities.max(axis=1)>=threshold

        # Predict and return those to-be-labeled data
        label_index = np.zeros_like(y, dtype=np.bool_)
        label_index[over_thresh & ~labeled]=True
        label_value = dnn.predict(X[over_thresh & ~labeled])
        return label_index, label_value


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
def data_prepare(dataset_path, n_train=60000, n_test=10000):
    image_size = 28

    num_images = n_train
    x_train_file = gzip.open(dataset_path+'/train-images-idx3-ubyte.gz','r')
    x_train_file.read(16)
    x_train_buf = x_train_file.read(image_size * image_size * num_images)
    x_train_all = np.frombuffer(x_train_buf, dtype=np.uint8).reshape(num_images, image_size * image_size) / 255.0

    y_train_file = gzip.open(dataset_path+'/train-labels-idx1-ubyte.gz','r')
    y_train_file.read(8)
    y_train_buf = y_train_file.read(num_images * 1)
    y_train_all = np.frombuffer(y_train_buf, dtype=np.uint8).reshape(num_images, )
    
    num_images = n_test
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

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,   required=True)
    parser.add_argument("--n_train", type=int,   default=6000)
    parser.add_argument("--n_test",  type=int,   default=10000)
    parser.add_argument("--model_cls", type=str, nargs='+', required=True)
    parser.add_argument("--model_slt", type=str, nargs='+', required=True)
    parser.add_argument("--ratios",    type=int, nargs='+', default=[10])
    parser.add_argument("--max_iterations", type=int, default=10)
    args = parser.parse_args()

    dataset_path    = args.dataset
    n_train_samples = args.n_train
    ratios          = args.ratios
    max_iterations  = args.max_iterations
    # result_file = 'self_training_mnist_ntrain'+str(n_train_samples)+'_label'+
    filename_res = 'SSL_result_mnist_ntrain%d_label%d_cls%s_slt%s.txt'

    dataset_all = data_prepare(dataset_path, n_train=n_train_samples, n_test=args.n_test)
    
    print(dataset_all['train_x'].shape)
    dataset = {}
    dataset['test_x'] = dataset_all['test_x']
    dataset['test_y'] = dataset_all['test_y']
    
    name_clfs = []
    if 'DNN'        in args.model_cls:
        name_clfs.append( ('DNN',        neural_network.MLPClassifier(hidden_layer_sizes=(512, 216), max_iter=500)))
    if 'SVM_rbf'    in args.model_cls:
        name_clfs.append( ('SVM_rbf',    svm.SVC(kernel='rbf', C=5, gamma=0.01, probability=True)))
    if 'SVM_linear' in args.model_cls:
        name_clfs.append( ('SVM_linear', svm.SVC(kernel='linear', C=0.1, probability=True)))
    if 'KNN5'       in args.model_cls:
        name_clfs.append( ('KNN5',       neighbors.KNeighborsClassifier(5, weights='uniform')))
    if 'logistics'  in args.model_cls:
        name_clfs.append( ('logistics',  linear_model.LogisticRegression(random_state=0, solver='sag', multi_class='multinomial', max_iter=500)))

    name_selectors = []
    if 'GMM' in args.model_slt:
        name_selectors.append( ('GMM', GMM() ))
    if 'DNN' in args.model_slt:
        name_selectors.append( ('DNN',       DNN_cluster()))

    Train_labeled_loss,     Train_unlabeled_loss,     Test_loss     = {}, {}, {}
    Train_labeled_accuracy, Train_unlabeled_accuracy, Test_accuracy = {}, {}, {}

    # # Check existed result
    # if os.path.isfile(result_file+'.pickle'):
    #     with open(result_file+'.pickle', 'rb') as file:
    #         results = pickle.load(file)
    #         Train_labeled_loss       = results['Train_labeled_loss']
    #         Train_unlabeled_loss     = results['Train_unlabeled_loss']
    #         Test_loss                = results['Test_loss']
    #         Train_labeled_accuracy   = results['Train_labeled_accuracy']
    #         Train_unlabeled_accuracy = results['Train_unlabeled_accuracy']
    #         Test_accuracy            = results['Test_accuracy']

    # iterate over classifiers
    for clf_name, clf in name_clfs:
        for slt_name, slt in name_selectors:
            # Train_labeled_loss[name], Train_unlabeled_loss[name], Test_loss[name] = [], [], []
            # Train_labeled_accuracy[name], Train_unlabeled_accuracy[name], Test_accuracy[name] = [], [], []

            for label_ratio in ratios:
                num_label_data = int( n_train_samples * label_ratio / 100 )
                dataset['train_labeled_x']   = dataset_all['train_x'][:num_label_data,]
                dataset['train_labeled_y']   = dataset_all['train_y'][:num_label_data]
                dataset['train_unlabeled_x'] = dataset_all['train_x'][num_label_data:n_train_samples,]
                dataset['train_unlabeled_y'] = dataset_all['train_y'][num_label_data:n_train_samples]

                method = StandardSelfTraining(clf_name, base_classifier=clf, base_cluster=slt, max_iterations=max_iterations)
                print("Method = ", method.__str__())
                train_labeled_losses, train_unlabeled_losses, test_losses, \
                   train_labeled_accuraciess, train_unlabeled_accuraciess, test_accuraciess = method.fit(dataset)


                single_result = {'Train_labeled_loss': train_labeled_losses,
                                 'Train_unlabeled_loss': train_unlabeled_losses,
                                 'Test_loss': test_losses,
                                 'Train_labeled_accuracy': train_labeled_accuraciess,
                                 'Train_unlabeled_accuracy': train_unlabeled_accuraciess,
                                 'Test_accuracy': test_accuraciess
                                 }
                with open(filename_res % (n_train_samples, label_ratio, clf_name, slt_name), "w") as f:
                    for item in single_result:
                        f.write(item+":"+str(single_result[item])+'\n')
            #     Train_labeled_loss[name].append(train_labeled_losses)
            #     Train_unlabeled_loss[name].append(train_unlabeled_losses)
            #     Test_loss[name].append(test_losses)
            #     Train_labeled_accuracy[name].append(train_labeled_accuraciess)
            #     Train_unlabeled_accuracy[name].append(train_unlabeled_accuraciess)
            #     Test_accuracy[name].append(test_accuraciess)

            # result_to_file = {'Train_labeled_loss': Train_labeled_loss,
            #                   'Train_unlabeled_loss': Train_unlabeled_loss,
            #                   'Test_loss': Test_loss,
            #                   'Train_labeled_accuracy': Train_labeled_accuracy,
            #                   'Train_unlabeled_accuracy': Train_unlabeled_accuracy,
            #                   'Test_accuracy': Test_accuracy}
            # with open(result_file+'.txt', "w") as file: file.write(str(result_to_file))
            # with open(result_file+'.pickle', "wb") as file: dump(result_to_file, file)
