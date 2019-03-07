import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str)
parser.add_argument("--n_samples",    nargs='+', type=int)
args = parser.parse_args()
dataset_path = '/home/chingyi/Datasets/mnist'
n_samples = [5000,60000]
dataset_path = args.dataset_path
n_samples = args.n_samples
# # Dataset preparation

# In[39]:

import csv
def dump_training_data( features, y_train, csv_name="xxx", dataset="cifar10" ):
    assert(features.shape[0]==y_train.shape[0])
    n_samples = y_train.shape[0]
    label_ratios = [0.1, 0.3, 0.5]
    for label_ratio in label_ratios:
        n_label   = int(n_samples*label_ratio)
        n_unlabel = int(n_samples*(1-label_ratio))
        case_name = dataset + '-total%d-label%d' % (n_samples, int(label_ratio*100)) + '-' + csv_name
        with open(case_name+'-labeled.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar=',', quoting=csv.QUOTE_MINIMAL)
            print(features[:n_label].shape, y_train[:n_label].shape)
            for ff, yy in zip(features[:n_label], y_train[:n_label]):
                spamwriter.writerow([yy]+ff.flatten().tolist())
        with open(case_name+'-unlabeled.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar=',', quoting=csv.QUOTE_MINIMAL)
            for ff, yy in zip(features[n_label:n_samples], y_train[n_label:n_samples]):
                spamwriter.writerow([yy]+ff.flatten().tolist())
def dump_testing_data( features, y_test, csv_name="xxx", dataset="cifar10" ):
    with open(dataset+'-test-all-'+csv_name+'.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar=',', quoting=csv.QUOTE_MINIMAL)
        for ff, yy in zip(features, y_test):
            spamwriter.writerow([yy]+ff.flatten().tolist())

# In[48]:


import numpy as np
import gzip
image_size = 28

num_images = 60000
x_train_file = gzip.open(dataset_path+'/train-images-idx3-ubyte.gz','r')
x_train_file.read(16)
x_train_buf = x_train_file.read(image_size * image_size * num_images)
x_train_all = np.frombuffer(x_train_buf, dtype=np.uint8).reshape(num_images, image_size, image_size, 1)

y_train_file = gzip.open(dataset_path+'/train-labels-idx1-ubyte.gz','r')
y_train_file.read(8)
y_train_buf = y_train_file.read(num_images * 1)
print(np.frombuffer(y_train_buf, dtype=np.uint8).shape)
y_train_all = np.frombuffer(y_train_buf, dtype=np.uint8).reshape(num_images, 1)


# In[47]:


num_images = 10000
x_test_file = gzip.open(dataset_path+'/t10k-images-idx3-ubyte.gz','r')
x_test_file.read(16)
x_test_buf = x_test_file.read(image_size * image_size * num_images)
x_test_all = np.frombuffer(x_test_buf, dtype=np.uint8).reshape(num_images, image_size, image_size, 1)

y_test_file = gzip.open(dataset_path+'/t10k-labels-idx1-ubyte.gz','r')
y_test_file.read(8)
y_test_buf = y_test_file.read(num_images * 1)
print(np.frombuffer(y_test_buf, dtype=np.uint8).shape)
y_test_all = np.frombuffer(y_test_buf, dtype=np.uint8).reshape(num_images, 1)


# In[49]:


for n_sample in n_samples:
    x_train = x_train_all[:n_sample]
    y_train = y_train_all[:n_sample].reshape(-1)
    dump_training_data( x_train, y_train, "raw", dataset='mnist')
x_test  = x_test_all
y_test  = y_test_all.reshape(-1)
print("y_test = ", y_test)
dump_testing_data(  x_test,  y_test,  "raw", dataset='mnist')


