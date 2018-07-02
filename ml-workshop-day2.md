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

## **Lab 1. Build your first Deep Learning Programm: Digit recognition**

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

#### **TIP** How to run the training script within Jupyter notebook not in a seperate training environment?

You can invoke _train_ function directly with the correct parameters.

_train(channel_input_dget_train_datairs, hyperparameters, hosts, num_gpus, **kwargs)_


````python
import mnist as pyfile
hyperparameters={'batch_size': 100,
                         'epochs': 10,
                         'learning_rate': 0.1,
                         'momentum': 0.9,
                         'log_interval': 100})

pyfile.train({'training': 'data/train/'},
    hyperparameters=hyper_parameters,
    hosts=['local'], num_gpus=0)
````
You can also use SageMaker local to run locally.

install sagemaker local
```python
!~/sample-notebooks/sagemaker-python-sdk/mxnet_gluon_mnist/setup.sh
```

You can now run Sagemaker local using:

```python
m = MXNet("mnist.py",
          role=role,
          train_instance_count=1,
          train_instance_type="local",
          hyperparameters={'batch_size': 100,
                         'epochs': 10,
                         'learning_rate': 0.1,
                         'momentum': 0.9,
                         'log_interval': 100})
m.fit(inputs)
```

You need to define _save_ function in your training script to save the trained model. SageMaker will call _save_ function with the return value of _train_ function. See the below sample code. Once the training job completes, the model file is being sent to S3 bucket.

```python
def save(net, model_dir):
    # save the model
    y = net(mx.sym.var('data'))
    y.save('%s/model.json' % model_dir)
    net.collect_params().save('%s/model.params' % model_dir)

```

Then, the function to make a prediction needs to be defined within a function, _transform_fn_.

````python
def transform_fn(net, data, input_content_type, output_content_type):
    """
    Transform a request using the Gluon model. Called once per request.

    :param net: The Gluon model.
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    # we can use content types to vary input/output handling, but
    # here we just assume json for both
    parsed = json.loads(data)
    nda = mx.nd.array(parsed)
    output = net(nda)
    prediction = mx.nd.argmax(output, axis=1)
    response_body = json.dumps(prediction.asnumpy().tolist()[0])
    return response_body, output_content_type

````

The prediction is also made through Amazon SageMaker Python SDK;
```python
response = predictor.predict(data)
```

> Refer to https://docs.aws.amazon.com/sagemaker/latest/dg/mxnet-training-inference-code-template.html for other functions to be defined in training scripts and inference scripts.

#### Training Result

After 10 epochs, the prediction accuracy on training data and validation data are given as 0.991183 and 0.973400 respectively. Your result will not be the same as this.

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

> NOTE: Refer to Gluon API (Basic Layers) at https://mxnet.incubator.apache.org/api/python/gluon.html

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

Use SageMaker Hyperparameter tuning jobs to tune the job hyper parameters.
https://github.com/aws/sagemaker-python-sdk 

> NOTE: Refer to Gluon API (Basic Layers) at https://mxnet.incubator.apache.org/api/python/gluon.html

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

> NOTE: Refer to Gluon API (Convolutional Layers, Pooling Layers) at https://mxnet.incubator.apache.org/api/python/gluon.html

> **Adam Optimizer on Gluon**
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

In this lab, we define different matric factorization models using Apache MXNet, and train them using MovieLens 100k dataset.

> **MovieLens 100K** (https://grouplens.org/datasets/movielens/100k/)
>
> Stable benchmark dataset. 100,000 ratings from 1000 users on 1700 movies. Released 4/1998.

### 3-1. Linear MF and Neural Network MF

> **Notebook:** recommenders/demo1-MF.ipynb
>
> **Python scripts:** recommenders/matrix_fact.py, recommenders/movielens_data.py

````python
def plain_net(k):
    # input
    user = mx.symbol.Variable('user')
    item = mx.symbol.Variable('item')
    score = mx.symbol.Variable('score')
    # user feature lookup
    user = mx.symbol.Embedding(data = user, input_dim = max_user, output_dim = k)
    # item feature lookup
    item = mx.symbol.Embedding(data = item, input_dim = max_item, output_dim = k)
    # predict by the inner product, which is elementwise product and then sum
    pred = user * item
    pred = mx.symbol.sum(data = pred, axis = 1)
    pred = mx.symbol.Flatten(data = pred)
    # loss layer
    pred = mx.symbol.LinearRegressionOutput(data = pred, label = score)
    return pred
````

````python
model = mx.model.FeedForward(
    ctx = ctx,
    symbol = network,
    num_epoch = num_epoch,
    optimizer = optimizer,
    learning_rate = learning_rate,
    wd = 1e-4,
    **opt_args
)

model.fit(X = train,
            eval_data = test,
            eval_metric = RMSE,
            batch_end_callback = [mx.callback.Speedometer(50, 500), my_callback]
            )
````
#### Challenge

Migrate this notebook and scripts to a training script which can be used by Amazon SageMaker.

> **Hint:**
>
> Move _plain_net()_ and _get_one_layer_mlp()_ to matrix_fact.py, and modify _train()_ function in matrix_fact.py. Refer to the previous Lab for train() modification.

### 3-2. Content-based recommender using DSSM

> **Notebook:** recommenders/demo3-dssm.ipynb
>
> **Python scripts:** recommenders/symbol_alexnet.py, recommenders/recotools.py

````python
def dssm_recommender(k):
    # input variables
    title = mx.symbol.Variable('title_words')
    image = mx.symbol.Variable('image')
    queries = mx.symbol.Variable('query_ngrams')
    user = mx.symbol.Variable('user_id')
    label = mx.symbol.Variable('label')

    # Process content stack
    image = alexnet.features(image, 256)
    title = recotools.SparseBagOfWordProjection(data=title, vocab_size=title_vocab, output_dim=k)
    title = mx.symbol.FullyConnected(data=title, num_hidden=k)
    content = mx.symbol.Concat(image, title)
    content = mx.symbol.Dropout(content, p=0.5)
    content = mx.symbol.FullyConnected(data=content, num_hidden=k)

    # Process user stack
    user = mx.symbol.Embedding(data=user, input_dim=max_user, output_dim=k)
    user = mx.symbol.FullyConnected(data=user, num_hidden=k)
    queries = recotools.SparseBagOfWordProjection(data=queries, vocab_size=ngram_dimensions, output_dim=k)
    queries = mx.symbol.FullyConnected(data=queries, num_hidden=k)
    user = mx.symbol.Concat(user,queries)
    user = mx.symbol.Dropout(user, p=0.5)
    user = mx.symbol.FullyConnected(data=user, num_hidden=k)

    # loss layer
    pred = recotools.CosineLoss(a=user, b=content, label=label)
    return pred

net1 = dssm_recommender(256)
mx.viz.plot_network(net1)
````

> **Deep matrix factorization using Apache MXNet** (A tutorial on how to use machine learning to build recommender systems.)
>
> https://www.oreilly.com/ideas/deep-matrix-factorization-using-apache-mxnet
