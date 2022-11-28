################################################################################
# MIT License
#
# Copyright (c) 2021 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2021-11-01
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils
import matplotlib.pyplot as plt
import torch


def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    #######################
    # END OF YOUR CODE    #
    #######################
    return conf_mat


def confusion_matrix_to_metrics(confusion_matrix, beta=1.):
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    pass
    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def evaluate_model(model, data_loader, num_classes=10):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # total predictions
    total = 0
    correct = 0
    # initialize dictionary
    metrics = {}

    for images, labels in data_loader:
        # flatten input
        images = np.reshape(images, (images.shape[0], -1))
        # get the scores of the forward pass
        scores = model.forward(images)
        # get the predicted labels
        predicted = np.argmax(scores, axis=1)
        # get all correct predictions
        total += images.shape[0]
        correct += (predicted == labels).sum().item()
  
    # calculate accuracy
    accuracy = correct/total
    metrics['accuracy'] = accuracy

    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics



def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    model = MLP(n_inputs=3*32*32, n_hidden=hidden_dims, n_classes=10)
    loss_module = CrossEntropyModule()

    # initialize return variables
    val_accuracies = []
    logging_dict = {}
    train_loss = []

    # best loss achieved in training
    best_loss = np.inf

    for epoch in tqdm(range(epochs)):
      # Running values
      loss = 0.0
      num_images = 0

      for images, labels in cifar10_loader['train']:
        # flatten input
        images = np.reshape(images, (images.shape[0], -1))
        # get the scores of the forward pass
        scores = model.forward(images)
        # forward pass
        current_loss = loss_module.forward(scores, labels)
        # backprop
        backprop_loss = loss_module.backward(scores, labels)
        model.backward(backprop_loss)

        # update parameters
        for layer in model.classnet:
          # update only linear layers
          try:
            layer.params['weight'] = layer.params['weight'] - lr * layer.grads['weight']
            layer.params['bias']   = layer.params['bias'] - lr * layer.grads['bias']
          # else skip
          except:
            continue

        # loss performance
        loss += current_loss
        num_images += images.shape[0]

      loss = loss/batch_size
      train_loss.append(loss)

      # validation
      metrics = evaluate_model(model=model, data_loader=cifar10_loader['validation'])
      val_accuracies.append(metrics['accuracy'])

      if epoch == 0:
        best_loss = current_loss
        best_model = deepcopy(model)
      else:
        if current_loss < best_loss:
          best_loss = current_loss
          best_model = deepcopy(model)

    # test the model
    metrics = evaluate_model(model=best_model, data_loader=cifar10_loader['test'])
    test_accuracy = metrics['accuracy']

    logging_dict['loss'] = train_loss
    logging_dict['val_ac'] = val_accuracies
    logging_dict['final_ac'] = test_accuracy

    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict

def plot_performance(logging_dict, name):

  fig = plt.figure(figsize=(14, 8))
  fig.suptitle(f'{name}: Numpy Performance', fontsize=18, y=0.95)
  
  plt.plot(np.arange(len(logging_dict['loss'])), logging_dict['loss'], 'o-', color='blue', linewidth='2', label='Train Loss')
  plt.plot(np.arange(len(logging_dict['val_ac'])), logging_dict['val_ac'], 'o-', color='orange', linewidth='2', label='Train Accuracy')

  # Add the test set performance
  plt.hlines(logging_dict['final_ac'], xmin=0, xmax=len(logging_dict['loss']), linestyles='dashed', label='Test Accuracy')
  plt.text(len(logging_dict['loss']), logging_dict['final_ac'], "{:.3f}".format(logging_dict['final_ac']))
  plt.legend()
  plt.xlabel('Epoch')
  plt.ylabel('Loss/Accuracy')
  plt.savefig(f'./data/{name}_numpy_performance.png', bbox_inches='tight')

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    model, val_accuracies, test_accuracy, logging_dict = train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    print(test_accuracy)
    plot_performance(logging_dict, 'MLP')
