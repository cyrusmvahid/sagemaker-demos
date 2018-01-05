from __future__ import print_function

import logging
import mxnet as mx
from mxnet import gluon, autograd,nd
from mxnet.gluon import nn
import numpy as np
import json
import time
from mxnet.gluon.data.vision import MNIST
import boto3
import gzip,struct
import os


logging.basicConfig(level=logging.DEBUG)

def yahoo():
    return('yoohoo!')
def yahoo1():
    return('yoohoo!')
# ------------------------------------------------------------ #
# Training methods                                             #
# ------------------------------------------------------------ #


def train(channel_input_dirs, hyperparameters, hosts, num_gpus, **kwargs):
    # SageMaker passes num_cpus, num_gpus and other args we can use to tailor training to
    # the current container environment, but here we just use simple cpu context.
    ctx = mx.gpu() if num_gpus > 0 else mx.cpu()
    print()
    print()
    print()
    print()
    print('**********************************************************************')
    print(channel_input_dirs)
    print(os.listdir(channel_input_dirs['training']))
    print('**********************************************************************')
    print()
    print()
    print()
    print()

    # retrieve the hyperparameters we set in notebook (with some defaults)
    batch_size = hyperparameters.get('batch_size', 100)
    epochs = hyperparameters.get('num_epochs', 10)
    learning_rate = hyperparameters.get('learning_rate', 0.1)
    momentum = hyperparameters.get('momentum', 0.9)
    log_interval = hyperparameters.get('log_interval', 100)
    num_dims = hyperparameters.get('num_dims')
    num_output = hyperparameters.get('num_outputs', 2)
    

    # load training and validation data
    training_dir = channel_input_dirs['training']
    train_data, test_data = load_data(training_dir)
    train_data_iter = get_train_data(train_data,batch_size)
    test_data_iter = get_test_data(test_data,batch_size)
    
    print(train_data[0].shape, train_data[0].shape, test_data[0].shape, test_data[0].shape)
    # define the network
    net = define_network(num_dims, num_output)
    print(net.collect_params())


    # Collect all parameters from net and its children, then initialize them.
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    # Trainer is for updating parameters with gradient.

    if len(hosts) == 1:
        kvstore = 'device' if num_gpus > 0 else 'local'
    else:
        kvstore = 'dist_device_sync' if num_gpus > 0 else 'dist_sync'

    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': learning_rate, 'momentum': momentum},
                            kvstore=kvstore)
    metric = mx.metric.RMSE()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    for epoch in range(epochs):
        # reset data iterator and metric at begining of epoch.
        for i, (data, label) in enumerate(train_data_iter):
            # Copy data to ctx if necessary
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            # Start recording computation graph with record() section.
            # Recorded graphs can then be differentiated with backward.
            with autograd.record():
                output = net(data)
                L = loss(output, label)
                L.backward()
            # take a gradient step with batch_size equal to data.shape[0]
            trainer.step(data.shape[0])

        test_accuracy = evaluate_accuracy(net, test_data_iter, ctx)
        train_accuracy = evaluate_accuracy(net, train_data_iter, ctx)
        print("Epoch %s, Train_acc %s, Test_acc %s" %
              (epoch, train_accuracy, test_accuracy))    
    return net


def save(net, model_dir):
    print()
    print()
    print()
    print()
    print('**********************************************************************')
    print('saving the model in {}'.format(model_dir))
    print('**********************************************************************')
    print()
    print()
    print()
    print()

    # save the model
    y = net(mx.sym.var('data'))
    y.save('%s/model.json' % model_dir)
    net.collect_params().save('%s/model.params' % model_dir)

def load_data(location):
    train_data = np.load(location+'/train/train_data.npy').astype(np.float32)
    train_label = np.load(location+'/train/train_label.npy').astype(np.float32)

    test_data = np.load(location+'/test/val_data.npy').astype(np.float32)
    test_label = np.load(location+'/test/val_label.npy').astype(np.float32)
    return (train_data, train_label), (test_data, test_label)
    
def define_network(num_inputs, num_outputs):
    net = nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(64, activation='relu'))
        net.add(gluon.nn.Dense(2))
    return net


def get_train_data(data,batch_size):
    return gluon.data.DataLoader(gluon.data.ArrayDataset(data[0], data[1]), 
                                        shuffle=True, batch_size=batch_size)


def get_test_data(data,batch_size):
    return gluon.data.DataLoader(gluon.data.ArrayDataset(data[0], data[1]), 
                                        shuffle=False, batch_size=batch_size)
  
def evaluate_accuracy(net, data_iterator, ctx):
    metric = mx.metric.RMSE()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        metric.update(preds=predictions, labels=label)
    return metric.get()

        

# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #

def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.

    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a Gluon network)
    """
    symbol = mx.sym.load('%s/model.json' % model_dir)
    outputs = mx.symbol.softmax(data=symbol, name='softmax_label')
    inputs = mx.sym.var('data')
    param_dict = gluon.ParameterDict('model_')
    net = gluon.SymbolBlock(outputs, inputs, param_dict)
    net.load_params('%s/model.params' % model_dir, ctx=mx.cpu())
    return net


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