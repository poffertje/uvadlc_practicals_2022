################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-14
################################################################################

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models
import os

from tqdm.auto import tqdm
from cifar100_utils import get_train_validation_set, get_test_set

MODELS_PATH = "./saved_models"

def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Get the pretrained ResNet18 model on ImageNet from torchvision.models
    model = models.resnet18(pretrained=True)
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Randomly initialize and modify the model's last layer for CIFAR100.
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)  # has requires_grad = True on initialization

    nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)
    nn.init.zeros_(model.fc.bias)

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name=None):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Load the datasets
    train_set, val_set = get_train_validation_set(data_dir)

    train_loader = data.DataLoader(train_set, batch_size, shuffle=True)
    val_loader = data.DataLoader(val_set, batch_size, shuffle=True)

    # Initialize the optimizer (Adam) to train the last layer of the model.
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad == True], lr=lr) 
    criterion = nn.CrossEntropyLoss()

    # Training loop with validation after each epoch. Save the best model.
    model.train() 
    best_accuracy = -np.inf

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0
        for images, labels in train_loader:

            # Move the training data on GPU
            images = images.to(device)
            labels = labels.to(device)

            # Run the model on the input data
            scores = model.forward(images)

            # Calculate loss
            current_loss = criterion(scores, labels)

            # Perform backpropagation
            optimizer.zero_grad()
            current_loss.backward(retain_graph=True)

            # Update the parameters
            optimizer.step()
            
            # Take the running average of the loss
            epoch_loss += current_loss.item()

            # Clear computation graph
            labels = labels.detach()
            scores = scores.detach()

        epoch_loss /= len(train_loader)

        # Validation loop
        val_accuracy = evaluate_model(model, val_loader, device)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss={epoch_loss:.3f}, Acc={val_accuracy:.3f}')

        if epoch == 0:
            best_accuracy = val_accuracy
            torch.save(model, checkpoint_name)
        else:
            if best_accuracy < val_accuracy:
                best_accuracy = val_accuracy
                torch.save(model, checkpoint_name)

    # Load the best model on val accuracy and return it.
    model = torch.load(checkpoint_name)

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set model to evaluation mode (Remember to set it back to training mode in the training loop)
    model.eval()
    correct, total = 0, 0

    # Loop over the dataset and compute the accuracy. Return the accuracy
    # Remember to use torch.no_grad().
    with torch.no_grad():
        for images, labels in data_loader:

            # Move the data on GPU
            images = images.to(device)
            labels = labels.to(device)

            # Run the model on the input data
            scores = model.forward(images)

            # Clear the computational graph
            images = images.detach().cpu()
            labels = labels.detach().cpu()

            # Get the predicted labels
            predicted = torch.argmax(scores, dim=1)

            # get all correct predictions
            correct += (predicted == labels).sum().item()
            total += labels.shape[0]
    
    accuracy = correct / total

    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def main(lr, batch_size, epochs, data_dir, seed, augmentation_name):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set the seed for reproducibility
    set_seed(seed)

    # Set the device to use for training
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Load the model
    model = get_model()

    # Get the augmentation to use
    

    # Train the model
    os.makedirs(MODELS_PATH, exist_ok=True)
    checkpoint_name = f'{MODELS_PATH}/resnet18_lr{lr}_batchsize{batch_size}_augmentation_{augmentation_name}.model'
    model = train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device)

    # Evaluate the model on the test set
    test_set = get_test_set(data_dir)
    test_loader = data.DataLoader(test_set, batch_size, shuffle=False)
    accuracy = evaluate_model(model, test_loader, device)
    print(f'FINAL PERFORMANCE:{accuracy:.3f}')

    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=123, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR100 dataset.')
    parser.add_argument('--augmentation_name', default=None, type=str,
                        help='Augmentation to use.')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
