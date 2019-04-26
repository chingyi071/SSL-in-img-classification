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


def get_label_mask( y_train, n_label ):#dataset_all,result_subdir, X_train, y_train, X_ten_label=args.nlabel):

    # Assign labels to a subset of inputs.
    num_classes = 10
    max_count = n_label // num_classes
    print("Keeping %d labels per class." % max_count)
    mask_train = np.zeros(len(y_train), dtype=np.bool)
    count = [0] * num_classes
    for i in range(len(y_train)):
        label = y_train[i]
        if count[label] < max_count:
            mask_train[i] = True
            count[label] += 1

    return mask_train

class StandardSelfTraining:
    def __init__(self, name, base_classifier, base_cluster, max_iterations=10, is_match=False):
        self.name = name
        self.base_classifier = base_classifier
        self.base_cluster    = base_cluster if base_cluster is not None else [base_classifier]
        self.max_iterations  = max_iterations
        self.label_indices   = []
        self.label_values    = []
        self.is_match        = is_match
        self.min_threshold   = 300

    def __str__(self):
        return "Classifier: " + self.name + "\nParameters: " + str(self.base_classifier.get_params())

    def fit(self, dataset):
        Xtrain = np.vstack((dataset['train_labeled_x'], dataset['train_unlabeled_x']))
        ytrain = np.hstack((dataset['train_labeled_y'], [99]*len(dataset['train_unlabeled_x'])))
        Xtest = dataset['test_x']
        ytest = dataset['test_y']
        X = np.copy(Xtrain)
        y = np.copy(ytrain) # copy in order not to change original data
        
        all_are_labeled = False
        iteration = 0
        train_labeled_losses, train_unlabeled_losses, test_losses = [], [], []
        train_labeled_accuraciess, train_unlabeled_accuraciess, test_accuraciess = [], [], []
        # Iterate until the result is stable or max_iterations is reached
        self.base_classifier.fit( dataset['train_labeled_x'], dataset['train_labeled_y'] )
        train_labeled_loss, train_labeled_error, train_labeled_accuracy = \
                            self.score(dataset['train_labeled_x'], dataset['train_labeled_y'])
        train_unlabeled_loss, train_unlabeled_error, train_unlabeled_accuracy = \
                            self.score(dataset['train_unlabeled_x'], dataset['train_unlabeled_y'])
        test_loss, test_error, test_accuracy = self.score(dataset['test_x'], dataset['test_y'])
        train_labeled_losses.       append(train_labeled_loss)
        train_unlabeled_losses.     append(train_unlabeled_loss)
        test_losses.                append(test_loss)
        train_labeled_accuraciess.  append(train_labeled_accuracy) 
        train_unlabeled_accuraciess.append(train_unlabeled_accuracy) 
        test_accuraciess.           append(test_accuracy)

        print('Before iteration #1, train_unlabeled_accuracy = '+str(train_unlabeled_accuracy)
              +', test_accuracy = '+str(test_accuracy))

        while not all_are_labeled and (iteration < self.max_iterations):
            print('Before iteration = #%d, train_unlabeled_accuracy = %.3f, test_accuracy = %.3f' % (iteration, train_unlabeled_accuracy, test_accuracy))
            
            cluster_index = iteration % len(self.base_cluster) #min(iteration, len(self.base_cluster)-1)
            print('Iteration #%d: Select cluster[%d/%d]: ' % (iteration, cluster_index, len(self.base_cluster)), type(self.base_cluster[cluster_index]))
            cluster = self.base_cluster[cluster_index] if self.base_cluster[cluster_index] is not None else self.base_classifier
            self._fit_iteration(X, y, cluster=cluster, is_match=self.is_match)
            all_are_labeled = (y != 99).all()

            train_labeled_loss, train_labeled_error, train_labeled_accuracy = \
                                self.score(dataset['train_labeled_x'], dataset['train_labeled_y'])
            train_unlabeled_loss, train_unlabeled_error, train_unlabeled_accuracy = \
                                self.score(dataset['train_unlabeled_x'], dataset['train_unlabeled_y'])
            test_loss, test_error, test_accuracy = self.score(dataset['test_x'], dataset['test_y'])
            print('After  iteration = #%d, train_unlabeled_accuracy = %.3f, test_accuracy = %.3f' % (iteration, train_unlabeled_accuracy, test_accuracy))
            
            # Record the loss
            train_labeled_losses.       append(train_labeled_loss)
            train_unlabeled_losses.     append(train_unlabeled_loss)
            test_losses.                append(test_loss)
            train_labeled_accuraciess.  append(train_labeled_accuracy) 
            train_unlabeled_accuraciess.append(train_unlabeled_accuracy) 
            test_accuraciess.           append(test_accuracy)
            iteration += 1

        return train_labeled_losses, train_unlabeled_losses, test_losses, \
               train_labeled_accuraciess, train_unlabeled_accuraciess, test_accuraciess
        

    def _fit_iteration(self, X, y, cluster, is_match=False):

        # Fit labeled data to base classifier
        clf = self.base_classifier
        labeled = (y != 99)
        clf.fit( X[labeled], y[labeled] )

        # Select to-be-labeled candidate
        label_index, label_value = cluster.selected_labeled(X,y,labeled)
        self.label_indices.append( label_index )
        self.label_values. append( label_value )

        # Return if no label selected by cluster
        if np.sum(label_index) == 0:
            return

        # Mask out (->unlabeled) the new label different with prediction from classfier
        y_new_label = clf.predict(X[label_index])
        match_predicted = y_new_label == label_value
        if is_match:
            label_value[~match_predicted] = 99
        # print("acc = ", np.sum(match_predicted), match_predicted.shape)

        # Assert every candidate in to-be-labeled is with y = 99
        for i, non_label in enumerate(label_index):
            assert(non_label == False or y[i] == 99)

        # Label new data
        print("before labeled size = ", len([x for x in (y != 99) if x == True]))
        y[label_index] = label_value

        # Dump number of labeled
        print("labeled size = ", len([x for x in (y != 99) if x == True]))

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

    def get_label_info(self):
        return self.label_indices, self.label_values

class GMM:
    def __init__( self,  th_label_assign=0.7, th_percentile=50 ):
        self.th_label_assign = th_label_assign
        self.th_percentile = th_percentile

    def gmm_analysis( self, X, y, labeled ):

        # New-labeled data selection
        # Calculate the clustering score
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=10, covariance_type='spherical')
        gmm.fit(X[labeled], y[labeled])
        clstr_score = gmm.score_samples(X)
        # print("prob varies from %f to %f" % (min(clstr_score), max(clstr_score)))
        
        # Filter the result over threshold
        # print("Number of unlabled data = ", clstr_score[~labeled].shape, "/", clstr_score.shape)
        predicted_index = gmm.predict(X)
        return predicted_index, clstr_score

    def label_assign( self, X, y, gmm_predicted ):
        clstr_count = np.zeros((10,10))
        for real_label in range(10):
            data_labeled_with_label_i = (y == real_label)
            index_in_gmm = gmm_predicted[data_labeled_with_label_i]
            for cluster_index in range(10):
                total = np.sum(index_in_gmm == cluster_index)
                clstr_count[real_label][cluster_index] = total
        # print("clstr_count = \n", clstr_count)

        max_in_count   = np.argmax(clstr_count, axis=0)
        over_threshold = np.max(clstr_count, axis=0)/np.sum(clstr_count, axis=0) > self.th_label_assign

        lut = [None]*10 # Indexed by cluster number get real label
        for i in range(10):
            if over_threshold[i] == True:
                lut[i] = max_in_count[i]
        # print("lut = ", lut)

        return lut

    def selected_labeled( self, X, y, labeled):
        gmm_predicted, clstr_score = self.gmm_analysis(X,y,labeled)
        lut = self.label_assign(X,y,gmm_predicted)

        to_be_labeled = -np.ones_like(y, dtype=np.int32)
        label_index = np.zeros_like(y, dtype=np.bool_)
        label_value = []
        import statistics
        for i in range(10):
            if lut[i] is not None:
                with_gmm_label_i = gmm_predicted==i
                # print("Number in cluster %d" % i, y[with_gmm_label_i].shape, "ranged from %d to %d" % (min(clstr_score[with_gmm_label_i]), max(clstr_score[with_gmm_label_i])))
#                stdev = statistics.stdev(clstr_score[with_gmm_label_i])
#                mean  = statistics.mean(clstr_score[with_gmm_label_i])
                value_percentile = np.percentile(clstr_score[with_gmm_label_i], self.th_percentile)
                over_per_cluster_thresh = clstr_score >= value_percentile

                num_of_new_label = sum(~labeled & over_per_cluster_thresh & with_gmm_label_i)
                if  num_of_new_label > 0:
                    to_be_labeled[~labeled & over_per_cluster_thresh & with_gmm_label_i] = lut[i]
                    label_index[~labeled & over_per_cluster_thresh & with_gmm_label_i] = True
                    label_value.extend([lut[i]]*num_of_new_label)
        label_index = to_be_labeled >= 0
        assert( np.sum(label_index) == to_be_labeled[label_index].shape[0] )
        return label_index, to_be_labeled[label_index]

class DNN_cluster:
    def __init__( self, th_label_assign, th_percentile=None ):
        self.th_label_assign = th_label_assign
        self.th_percentile = th_percentile

    def selected_labeled(self, X, y, labeled):
        import statistics

        # Make sure @labeled is correct
        for i, is_label in enumerate(labeled):
            assert(is_label == True or y[i] == 99)

        # Define a DNN classfier
        dnn = neural_network.MLPClassifier(hidden_layer_sizes=(256,256,256), max_iter=500)

        # Fit to data and get the prob for unlabeled data
        y_int = [int(yy) for yy in y[labeled]]
        dnn.fit(X[labeled],y_int)
        probabilities = dnn.predict_proba(X)

        prob_max = np.max(probabilities, axis=1)
        # stdev = statistics.stdev(prob_max[~labeled])
        # mean  = statistics.mean(prob_max[~labeled])
        if self.th_percentile is not None:
            mhigh = np.percentile(prob_max[~labeled], self.th_percentile) 

        else: mhigh = 0
        threshold = max(self.th_label_assign, mhigh)
        over_thresh = prob_max >= threshold
        print(over_thresh)

        # # Select the unlabeled data with prob greater than a threshold
        # threshold = min(0.8, probabilities[~labeled, :].max())
        # over_thresh = probabilities.max(axis=1)>=threshold

        # Predict and return those to-be-labeled data
        label_index = np.zeros_like(y, dtype=np.bool_)
        label_index[over_thresh & ~labeled]=True
        if X[over_thresh & ~labeled].shape[0]>0:
            label_value = dnn.predict(X[over_thresh & ~labeled])
        else: label_value = np.array([])
        # print("label_value = ", label_value)
        return label_index, label_value

class inherit_DNN_cluster(neural_network.MLPClassifier):
    def __init__(self, hidden_layer_sizes, max_iter, th_label_assign):
        super(inherit_DNN_cluster,self).__init__( hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter )
        self.th_label_assign = th_label_assign

    def selected_labeled(self, X, y, labeled):
        import statistics

        # Make sure @labeled is correct
        for i, is_label in enumerate(labeled):
            assert(is_label == True or y[i] == 99)

        # Fit to data and get the prob for unlabeled data
        y_int = [int(yy) for yy in y[labeled]]
        self.fit(X[labeled],y_int)
        probabilities = self.predict_proba(X)

        prob_max = np.max(probabilities, axis=1)
        # stdev = statistics.stdev(prob_max[~labeled])
        # mean  = statistics.mean(prob_max[~labeled])
        mhigh = statistics.median_high(prob_max[~labeled])
        threshold = max( self.th_label_assign, mhigh)
        over_thresh = prob_max >= threshold

        # # Select the unlabeled data with prob greater than a threshold
        # threshold = min(0.8, probabilities[~labeled, :].max())
        # over_thresh = probabilities.max(axis=1)>=threshold

        # Predict and return those to-be-labeled data
        label_index = np.zeros_like(y, dtype=np.bool_)
        label_index[over_thresh & ~labeled]=True
        label_value = self.predict(X[over_thresh & ~labeled])
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
    
    # indice = np.zeros_like(y_train_all)
    # assert( len(indice.shape) == 1 )
    # arg = np.arange(6000)
    # for i in range(10):
    #     indice[arg[y_train_all==i][:10]] = True
    # print("indice = ", indice, sum(indice))

    # xxx
    
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


def merge_result( single_result ):
    for item in single_result:
        if 'New_label' not in item:
            avg = np.average( single_result[item], axis=0 )
            avg_result[item] = avg

    new_labels_corrected = []
    new_labels_total = []
    correctness = np.zeros( single_result['New_label_value'].shape[:2] )
    label_total = np.zeros( single_result['New_label_value'].shape[:2] )
    for i, (one_expr_label_indices, one_expr_label_total) in enumerate(zip( single_result['New_label_correct'], single_result['New_label_value'])):                        
        for j, (label_index, label_value) in enumerate(zip( one_expr_label_indices, one_expr_label_total )):
            correctness[i][j] = np.sum( label_value==groundtruth[label_index])
            label_total[i][j] = label_value.shape[0]

    avg_result["New_label_correct"] = np.average(correctness, axis=0)
    avg_result["New_label_total"]   = np.average(label_total, axis=0)
    niter = avg_result["New_label_total"].shape[0]
    return niter, avg_result


# ================================================================================== #
#                                       main                                         #
# ================================================================================== #
if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,   required=True)
    parser.add_argument("--n_label", type=int,   default=600)
    parser.add_argument("--n_train", type=int,   default=6000)
    parser.add_argument("--n_test",  type=int,   default=10000)
    parser.add_argument("--model_cls", type=str, nargs='+', required=True)
    parser.add_argument("--model_slt", type=str, nargs='+', required=False)
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--model_slt_gmm", action='store_true')
    parser.add_argument("--model_slt_gmm_lbasn", type=int, nargs='+', default=[60])
    parser.add_argument("--model_slt_gmm_pcntl", type=int, nargs='+', default=[75])
    parser.add_argument("--model_slt_dnn", action='store_true')
    parser.add_argument("--model_slt_dnn_lbasn", type=int, nargs='+', default=[90])
    parser.add_argument("--model_slt_dnn_noniv", action='store_true')
    parser.add_argument("--model_slt_dnn_mhigh", action='store_true')
    parser.add_argument("--is_match", help="Selected data should be matched with classifier prediction", action='store_true')
    parser.add_argument("--niter_per_config", type=int, default=10)
    parser.add_argument("--log_path", type=str, default=".")
    args = parser.parse_args()

    dataset_path    = args.dataset
    n_train_samples = args.n_train
    n_label_samples = args.n_label
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
        name_clfs.append( ('DNN',        neural_network.MLPClassifier(hidden_layer_sizes=(256,256,256), max_iter=500)))
    if 'iDNN'        in args.model_cls:
        name_clfs.append( ('DNN',        inherit_DNN_cluster(hidden_layer_sizes=(256,256,256), max_iter=500)))
    if 'SVM_rbf'    in args.model_cls:
        name_clfs.append( ('SVM_rbf',    svm.SVC(kernel='rbf', C=5, gamma=0.01, probability=True)))
    if 'SVM_linear' in args.model_cls:
        name_clfs.append( ('SVM_linear', svm.SVC(kernel='linear', C=0.1, probability=True)))
    if 'KNN5'       in args.model_cls:
        name_clfs.append( ('KNN5',       neighbors.KNeighborsClassifier(5, weights='uniform')))
    if 'logistics'  in args.model_cls:
        name_clfs.append( ('logistics',  linear_model.LogisticRegression(random_state=0, solver='sag', multi_class='multinomial', max_iter=500)))

    name_selectors = []

    if args.model_slt_gmm:
        for label_assign in args.model_slt_gmm_lbasn:
            for percentile in args.model_slt_gmm_pcntl:
                model_name = "GMM%d%d" % (label_assign, percentile)
                if args.is_match: model_name = model_name + '-match'
                name_selectors.append( (model_name, [GMM(th_label_assign=label_assign/100, th_percentile=percentile)]))
    else: print("No GMM model")

    if args.model_slt_dnn:
        for label_assign in args.model_slt_dnn_lbasn:
            model_name = "%s" + "-DNN%d" % label_assign
            if args.is_match: model_name = model_name + '-match'
            if not args.model_slt_dnn_noniv:
                specific_model_name = model_name % "naive"
                name_selectors.append( (specific_model_name, [DNN_cluster(th_label_assign=label_assign/100)]))
            if args.model_slt_dnn_mhigh:
                specific_model_name = model_name % "mhigh"
                name_selectors.append( (specific_model_name, [DNN_cluster(th_label_assign=label_assign/100, th_percentile=75)]))
    else: print("No DNN model")

    print("Data distribution: label(%d), train(%d), test(%d)" % (n_train_samples, n_label_samples, args.n_test))
    print("To-be-trained selectors: ")
    for i, model in enumerate(name_selectors):
        print("#%-2d: " % i, model)

    print("log path will be saved into " + args.log_path)
    print("====================")

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
        for slt_name, slts in name_selectors:
            # Train_labeled_loss[name], Train_unlabeled_loss[name], Test_loss[name] = [], [], []
            # Train_labeled_accuracy[name], Train_unlabeled_accuracy[name], Test_accuracy[name] = [], [], []
            print("Selector = ", slt_name)
            train_labeled_losses = []
            train_unlabeled_losses = []
            test_losses = []
            train_labeled_accuraciess = []
            train_unlabeled_accuraciess = []
            test_accuraciess = []
            label_indices = []
            label_values = []
            
            for _ in range(args.niter_per_config):
                label_mask = get_label_mask( y_train=dataset_all['train_y'][:n_train_samples], n_label=n_label_samples )
                dataset['train_labeled_x']   = dataset_all['train_x'][:n_train_samples][label_mask]
                dataset['train_labeled_y']   = dataset_all['train_y'][:n_train_samples][label_mask]
                dataset['train_unlabeled_x'] = dataset_all['train_x'][:n_train_samples][~label_mask]
                dataset['train_unlabeled_y'] = dataset_all['train_y'][:n_train_samples][~label_mask]

                method = StandardSelfTraining(clf_name, base_classifier=clf, base_cluster=slts, max_iterations=max_iterations, is_match=args.is_match)
                print("Method = ", method.__str__())
                train_labeled_loss, train_unlabeled_loss, test_loss, \
                   train_labeled_acc, train_unlabeled_acc, test_acc = method.fit(dataset)
                label_index, label_value = method.get_label_info()

                train_labeled_losses.append( train_labeled_loss )
                train_unlabeled_losses.append( train_unlabeled_loss )
                test_losses.append( test_loss )
                train_labeled_accuraciess.append( train_labeled_acc )
                train_unlabeled_accuraciess.append( train_unlabeled_acc )
                test_accuraciess.append( test_acc )
                label_indices.append(label_index)
                label_values.append(label_value)


            single_result = {'Train_labeled_loss':       np.array(train_labeled_losses),
                             'Train_unlabeled_loss':     np.array(train_unlabeled_losses),
                             'Test_loss':                np.array(test_losses),
                             'Train_labeled_accuracy':   np.array(train_labeled_accuraciess),
                             'Train_unlabeled_accuracy': np.array(train_unlabeled_accuraciess),
                             'Test_accuracy':            np.array(test_accuraciess),
                             'New_label_correct':      np.array(label_indices),
                             'New_label_value':          np.array(label_values),
                             }

            groundtruth = np.hstack((dataset['train_labeled_y'], dataset['train_unlabeled_y']))
            
            filename = args.log_path + '/' + "SSL-clf-%s-slt-%s-label%d.txt" % (clf_name, slt_name, n_label_samples)
            avg_result = {}
            with open(filename, 'w') as f:

                niter, avg_result = merge_result(single_result)
                for item in avg_result:
                    f.write(item + ' ')
                f.write('\n')

                for i in range(niter):
                    for item in avg_result:
                        f.write(str(avg_result[item][i])+' ')
                    f.write('\n')
