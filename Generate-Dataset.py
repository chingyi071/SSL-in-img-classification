#!/usr/bin/env python
# coding: utf-8

# ## Dependency

# In[1]:


from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", help="Path of datapath", type=str)
parser.add_argument("--n_sample",     help="Number of sample", type=int)
args = parser.parse_args()


n_samples = [args.n_sample]
# ## Model Definition

# In[2]:


class cifar10vgg:
    def __init__(self,train=True):
        self.num_classes = 10
        self.weight_decay = 0.0005
        self.x_shape = [32,32,3]

        self.model = self.build_model()
        if train:
            self.model = self.train(self.model)
        else:
            self.model.load_weights('cifar10vgg.h5')


    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model

    def normalize_production(self,x):
        mean = 120.707
        std = 64.15
        return (x-mean)/(std+1e-7)

    def predict(self,x,normalize=True,batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x,batch_size)
    


# In[3]:


model = cifar10vgg(train=False)


# ### Layer Information

# In[4]:


# outputs = [layer.output for layer in model.model.layers]          # all layer outputs
# functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]    # evaluation functions
# layer_outs = [func([test, 1.]) for func in functors]
# for index, layer_out in enumerate(layer_outs):
#     print("Layer #%d" % index, layer_out[0].shape)


# # CIFAR-10 Dataset
# In[16]:


import torchvision
import numpy as np
dataset_test  = torchvision.datasets.CIFAR10( args.dataset_path, train=False, download=True)
dataset_train = torchvision.datasets.CIFAR10( args.dataset_path, train=True, download=True)

#    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train_all_raw = np.array([np.array(x) for x,_ in dataset_train])
y_train_all     = np.array([np.array(y) for _,y in dataset_train])
x_test_raw  = np.array([np.array(x) for x,_ in dataset_test] )
y_test      = np.array([np.array(y) for _,y in dataset_test] )


# In[17]:


dataset_test, dataset_train


# ### Normalization of image

# In[18]:


def normalize_production(x):
    mean = 120.707
    std = 64.15
    return (x-mean)/(std+1e-7)
x_train_all = normalize_production(x_train_all_raw)
x_test      = normalize_production(x_test_raw)


# # Dataset preparation

# In[19]:


import csv
def dump_training_data( features, y_train, csv_name="xxx" ):
    assert(features.shape[0]==y_train.shape[0])
    n_samples = y_train.shape[0]
    label_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    for label_ratio in label_ratios:
        n_label   = int(n_samples*label_ratio)
        n_unlabel = int(n_samples*(1-label_ratio))
        case_name = 'cifar10-total%d-label%d' % (n_samples, int(label_ratio*100)) + '-' + csv_name
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
def dump_testing_data( features, y_test, csv_name="xxx" ):
    with open('cifar10-test-all-'+csv_name+'.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar=',', quoting=csv.QUOTE_MINIMAL)
        for ff, yy in zip(features, y_test):
            spamwriter.writerow([yy]+ff.flatten().tolist())


# In[20]:


for n_sample in n_samples:
    x_train = x_train_all[:n_sample]
    y_train = y_train_all[:n_sample]
    dump_training_data( x_train, y_train, "normalized")
# dump_testing_data(  x_test,  y_test,  "normalized")


# In[21]:


from sklearn.decomposition import PCA
for n_sample in n_samples:
    x_train = x_train_all[:n_sample]
    y_train = y_train_all[:n_sample]
    x_train_flatten = x_train.reshape(x_train.shape[0],-1)
    x_test_flatten  = x_test. reshape(x_test .shape[0],-1)

    for n_components in [10,100,500]:
        if n_sample > n_components:
            pca = PCA(n_components=n_components).fit(x_train_flatten)
            train_features = pca.transform(x_train_flatten)
            dump_training_data( train_features, y_train, "pca"+str(n_components))
            # PCA transform is dependent with data size. Put testing set inside loop
            test_features  = pca.transform(x_test_flatten)
            dump_testing_data(  test_features,  y_test, "pca"+str(n_components))


# ## Prediction of VGG-16

# ### Feature extraction of VGG-16

# In[23]:


submodel = Sequential(model.model.layers[:-6])
for n_sample in n_samples:
    x_train = x_train_all[:n_sample]
    y_train = y_train_all[:n_sample]
    train_features = submodel.predict(x_train, 50)
    dump_training_data( train_features, y_train, "vgg16")

# # VGG-16 inference is independent with data size. Isolates testing set
# test_features = submodel.predict(x_test, x_test.shape[0])
# dump_testing_data(  features, y_test,  "vgg16")


# In[ ]:


import csv
with open('cifar10-test-all-pca10.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar=',')
    content = [row for row in spamreader]
    print(len(content),'*', len(content[0]))

