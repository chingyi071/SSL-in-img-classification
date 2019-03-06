import torchvision
import numpy as np

# Argument parser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str)
parser.add_argument("--n_samples",    nargs='+', type=int)
parser.add_argument("--n_color_lvls", nargs='+', type=int, default=[16])
args = parser.parse_args()
dataset_path = args.dataset_path
n_samples    = args.n_samples
n_clr_lvls   = args.n_color_lvls

# Load dataset
dataset_test  = torchvision.datasets.CIFAR10( args.dataset_path, train=False, download=True)
dataset_train = torchvision.datasets.CIFAR10( args.dataset_path, train=True, download=True)
x_train_all_raw = np.array([np.array(x) for x,_ in dataset_train])
y_train_all     = np.array([np.array(y) for _,y in dataset_train])
x_test_raw  = np.array([np.array(x) for x,_ in dataset_test] )
y_test      = np.array([np.array(y) for _,y in dataset_test] )

# Utility function definition
import csv
def dump_training_data( features, y_train, csv_name="xxx", dataset="cifar10" ):
    assert(features.shape[0]==y_train.shape[0])
    n_samples = y_train.shape[0]
    label_ratios = [0.1, 0.3, 0.5]
    for label_ratio in label_ratios:
        n_label   = int(n_samples*label_ratio)
        n_unlabel = int(n_samples*(1-label_ratio))
        case_name = dataset + '-total%d-label%d' % (n_samples, int(label_ratio*100)) + '-' + csv_name
        print("Generate "+case_name+'-labeled.csv')
        with open(case_name+'-labeled.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar=',', quoting=csv.QUOTE_MINIMAL)
            for ff, yy in zip(features[:n_label], y_train[:n_label]):
                spamwriter.writerow([yy]+ff.flatten().tolist())
        print("Generate "+case_name+'-unlabeled.csv')
        with open(case_name+'-unlabeled.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar=',', quoting=csv.QUOTE_MINIMAL)
            for ff, yy in zip(features[n_label:n_samples], y_train[n_label:n_samples]):
                spamwriter.writerow([yy]+ff.flatten().tolist())
def dump_testing_data( features, y_test, csv_name="xxx", dataset="cifar10" ):
    print("Generate "+dataset+'-test-all-'+csv_name+'.csv')
    with open(dataset+'-test-all-'+csv_name+'.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar=',', quoting=csv.QUOTE_MINIMAL)
        for ff, yy in zip(features, y_test):
            spamwriter.writerow([yy]+ff.flatten().tolist())

for n_clr_lvl in n_clr_lvls:
    n_clr_step = int(256/n_clr_lvl)
    for n_sample in n_samples:
        y_train = y_train_all[:n_sample]
        train_features_list = []
        for img in x_train_all_raw[:n_sample]:
            train_features_list.append(np.concatenate([np.histogram(img[:,:,c], bins=np.arange(0,256+n_clr_step,n_clr_step))[0] for c in range(3)]))
        train_features = np.array(train_features_list)
        dump_training_data( train_features, y_train, "hist"+str(n_clr_lvl))

    test_features_list = []
    for img in x_test_raw:
        train_features_list.append(np.concatenate([np.histogram(img[:,:,c], bins=np.arange(0,256+n_clr_step,n_clr_step))[0] for c in range(3)]))
    test_features = np.array(test_features_list)
    dump_testing_data( train_features, y_test, "hist"+str(n_clr_lvl))


