# Dataset Generation
## MNIST
### Source
The dataset can be downloaded from here: http://yann.lecun.com/exdb/mnist/

### Usage
```
python3 mnist_gen.py  --dataset_path <path/of/dataset> --n_samples <number/of/samples>
```
ex. python3 mnist_gen.py  --dataset_path /home/chingyi/Datasets/mnist/ --n_samples 6000 60000

```
python3 self_training.py --n_train=6000 --model_cls DNN --model_slt DNN GMM --dataset ~/Datasets/mnist/
cat SSL_result_mnist_ntrain6000_label10_clsDNN_sltGMM.txt
```
