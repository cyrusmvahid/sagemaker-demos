# **ML Workshop Day 2**

## **How to set up Lab environment**

1. Launch Jupyter notebook from Amazon SageMaker
2. Open Jupyter notebook, and open "terminal" by choosing "_New > Terminal_"
3. Make a clone of git repository

```{r, engine='bash', count_lines}
cd SageMaker
git clone https://github.com/cyrusmvahid/sagemaker-demos.git
```
---

## **Amazon SageMaker Python SDK**
Amazon SageMaker Python SDK is an open source library for training and deploying machine-learned models on Amazon SageMaker. Using the SDK, you can;

- train and deploy models using popular deep learning frameworks: Apache MXNet and TensorFlow
- train and deploy models with algorithms provided by Amazon
- train and host models using your own algorithms built into SageMaker-compatible Docker containers

### Git repository: https://github.com/aws/sagemaker-python-sdk
### API Doc
- https://readthedocs.org/projects/sagemaker/
- http://sagemaker.readthedocs.io/en/latest/

---

## **Lab 1. Build your first Deep Learning Programme: Digit recognition**

### 1-1. MNIST handwritten digit predection using MLP 

> **Notebook** : mxnet_gluon_mlp/MNIST/mnist_with_gluon.ipynb
>
> **Training/Prediction Code** : mxnet_gluon_mlp/MNIST/mnist.py

Amazon SageMaker will call **_train_** function for training and pass parameters for training.

```python
def train(channel_input_dirs, hyperparameters, hosts, num_gpus, **kwargs):
    ......
    batch_size = hyperparameters.get('batch_size', 100)
    epochs = hyperparameters.get('epochs', 10)
    learning_rate = hyperparameters.get('learning_rate', 0.1)
    momentum = hyperparameters.get('momentum', 0.9)
    log_interval = hyperparameters.get('log_interval', 100)
    ......
    training_dir = channel_input_dirs['training']
    ......
    if len(hosts) == 1:
        kvstore = 'device' if num_gpus > 0 else 'local'
    else:
        kvstore = 'dist_device_sync' if num_gpus > 0 else 'dist_sync'
    ......
```

The below is the function to define Neural Network architecture.

```python
def define_network():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(128, activation='relu'))
        net.add(nn.Dense(64, activation='relu'))
        net.add(nn.Dense(10))
    return net
```
The notebook code run the train code using Amazon SageMaker Python SDK;

```python
inputs = sagemaker_session.upload_data(path='data', key_prefix='data/mnist')

m = MXNet("mnist.py", 
          role=role, 
          train_instance_count=1, 
          train_instance_type="ml.c4.xlarge",
          hyperparameters={'batch_size': 100, 
                         'epochs': 10, 
                         'learning_rate': 0.1, 
                         'momentum': 0.9, 
                         'log_interval': 100})
m.fit(inputs)
```

The prediction is also made through Amazon SageMaker Python SDK;
```python
response = predictor.predict(data)
```

#### Training Result 

After 10 epoch, the accuracy of training data and validation data are given as 0.991183 and 0.973400 respectively. Your result will not be the same as this.

````
[Epoch 9] Training: accuracy=0.991183
[Epoch 9] Validation: accuracy=0.973400
````

#### **Challenges**

Modify the belows either jupyter notebook or python code and find the training performance;

1) hyper-parameter
2) activation functions
3) number of instances or instance type

Also, add more Dense (or fully connected) layers and observe if you can get better models.



#### **Question**

Do you think whether this model (MLP or Multilayer Perceptron) can make a good prediction if the digit is not positioned in the center?

If not, what is the reason and how to build a model to make a good prediction regardless the position of digits?

### 1-2. Fashion-MNIST classification using MLP

> **Notebook** : mxnet_gluon_mlp/FMNIST/fmnist_with_gluon.ipynb
>
> **Training/Prediction Code** : mxnet_gluon_mlp/FMNIST/fmnist.py

#### Training Result

With Fashion MNIST dataset, the MLP gives lower accuracy than the model for MNIST dataset. What is the reason of this?

````
[Epoch 9] Training: accuracy=0.894683
[Epoch 9] Validation: accuracy=0.885300
````

#### Challenge

Add more Dense (or fully connected) layers and observe if you can get better accuracy.

---
## **Lab 2. Build your first CNN**

### 2-1. MNIST classification using CNN

> **Notebook** : mxnet_gluon_cnn/MNIST/MNIST_with_gluon-cnn.ipynb
>
> **Training/Prediction Code** : mxnet_gluon_cnn/MNIST/mnist_cnn.py


````python
def define_network():
    num_outputs = 10
    num_fc = 512
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))            
        net.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        # The Flatten layer collapses all axis, except the first one, into one axis.
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(num_fc, activation="relu"))
        net.add(gluon.nn.Dense(num_outputs))
    return net
````

#### Training Result

Compare the accuracy of training and validation with accuracy of MLP result.

````
[Epoch 9] Training: accuracy=0.996383
[Epoch 9] Validation: accuracy=0.990500
````

#### Task

Write digits at non-center location and check the prediction. Even rotate digits.

If the prediction is not correct, what is the problem and how to address it?

2-2. Fashion-MNIST classification using CNN

> Notebook : mxnet_gluon_cnn/FMNIST/FMNIST_with_gluon-cnn.ipynb

#### Training Result

Compare the accuracy of training and validation with accuracy of MLP result.

````
[Epoch 9] Training: accuracy=0.916150
[Epoch 9] Validation: accuracy=0.905500
````

#### Challenge

Make any modification to increase the accuracy.

- Adding more Convolutional Layers
- Changing the optimizer (the code uses SGD)

> Adam Optimizer on Gluon
> 
> http://gluon.mxnet.io/chapter06_optimization/adam-gluon.html
> 
> 
> ````
> [Epoch 9] Training: accuracy=0.904250
> [Epoch 9] Validation: accuracy=0.886200
> ````

---


## **Lab 3. Build your first Recommender System**

https://www.oreilly.com/ideas/deep-matrix-factorization-using-apache-mxnet

### 3-1. Linear MF and Neural Network MF

> Notebook: demo1-MF.ipynb




### 3-2.

### 3-3.