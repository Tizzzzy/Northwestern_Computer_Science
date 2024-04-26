import sys
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
from FFNN import FFNN, FFNN_REG

def read_mnist(file_name):
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)
        
def show_mnist(file_name,mode):
    
    data_set = read_mnist(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')
                   
def read_insurability(file_name):
    
    count = 0
    data = []
    with open(file_name,'rt') as f:
        for line in f:
            if count > 0:
                line = line.replace('\n','')
                tokens = line.split(',')
                if len(line) > 10:
                    x1 = float(tokens[0])
                    x2 = float(tokens[1])
                    x3 = float(tokens[2])
                    if tokens[3] == 'Good':
                        cls = 0
                    elif tokens[3] == 'Neutral':
                        cls = 1
                    else:
                        cls = 2
                    data.append([[cls],[x1,x2,x3]])
            count = count + 1
    return(data)


def data_preprocess_three(records):
    """
    Helper function that pre-processes the input records for 
    insurability dataset to perform classification
    """
    features = [item[1] for item in records]
    labels = [item[0][0] for item in records]
    
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels)
    return features, labels


def data_preprocess_mnist(records):
    """
    Helper function that pre-processes the input records for 
    MNIST dataset to perform classification
    """
    features = [item[1] for item in records]
    labels = [item[0] for item in records]
    
    features = [[int(string) for string in item] for item in features]
    labels = [int(string) for string in labels]
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels)
    return features, labels
               
    
def classify_insurability():
    train = read_insurability('three_train.csv')
    valid = read_insurability('three_valid.csv')
    test = read_insurability('three_test.csv')
    
    # insert code to train simple FFNN and produce evaluation metrics
    features_train, labels_train = data_preprocess_three(train)
    features_val, labels_val = data_preprocess_three(valid)
    features_test, labels_test = data_preprocess_three(test)
    train_dataset = TensorDataset(features_train, labels_train)
    val_dataset = TensorDataset(features_val, labels_val)
    test_dataset = TensorDataset(features_test, labels_test)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = FFNN(3, 2, 3) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    num_epochs = 100  # Set the number of epochs
    best_val_loss = float('inf')
    y_predicted = []

    # create lists to store average losses
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            # Forward pass
            logits = model(X_batch)

            # Compute loss
            loss = criterion(logits, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                logits_val = model(X_val)
                loss_val = criterion(logits_val, y_val)
                total_val_loss += loss_val.item()


        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

    # Evaluate on Test Set
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for X_test, y_test in test_loader:
            logits_test = model(X_test)
            loss_test = criterion(logits_test, y_test)
            total_test_loss += loss_test.item()

            prob = model.softmax(logits_test)
            # Make the prediction:
            predicted_class = torch.argmax(prob, dim=1)
            predicted_class = predicted_class.tolist()

            y_predicted.append(predicted_class)

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    print("======================================= Classification Finished =======================================")
    y_predicted = [item for sublist in y_predicted for item in sublist]
    print(classification_report(labels_test, y_predicted))
    
def classify_mnist():
    
    train = read_mnist('mnist_train.csv')
    valid = read_mnist('mnist_valid.csv')
    test = read_mnist('mnist_test.csv')
    
    # insert code to train a neural network with an architecture of your choice
    # (a FFNN is fine) and produce evaluation metrics
    features_train, labels_train = data_preprocess_mnist(train)
    features_val, labels_val = data_preprocess_mnist(valid)
    features_test, labels_test = data_preprocess_mnist(test)
    
    # Convert to 0-1 scale to reduce computational cost
    features_train /= 255.0
    features_val /= 255.0
    features_test /= 255.0
    
    # Convert to PyTorch Tensor dataset
    train_dataset = TensorDataset(features_train, labels_train)
    val_dataset = TensorDataset(features_val, labels_val)
    test_dataset = TensorDataset(features_test, labels_test)
    
    # Non-lazily load in dataset so that we can split the data into batches, and shuffle the data
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Construct FFNN using the defined class. MNIST dataset has 784 pixels, and 10 digits (0-9)
    # Choose 128 hidden layers for more capacity
    model = FFNN(784, 128, 10) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    num_epochs = 300  # Set the number of epochs
    y_predicted = []

    # create lists to store average losses
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            # Forward pass
            logits = model(X_batch)

            # Compute loss
            loss = criterion(logits, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                logits_val = model(X_val)
                loss_val = criterion(logits_val, y_val)
                total_val_loss += loss_val.item()


        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")
        
    # Evaluate on Test Set
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for X_test, y_test in test_loader:
            logits_test = model(X_test)
            loss_test = criterion(logits_test, y_test)
            total_test_loss += loss_test.item()

            prob = model.softmax(logits_test)
            # Make the prediction:
            predicted_class = torch.argmax(prob, dim=1)
            predicted_class = predicted_class.tolist()

            y_predicted.append(predicted_class)

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    print("======================================= Classification Finished =======================================")
    
    y_predicted = [item for sublist in y_predicted for item in sublist]
    print(classification_report(labels_test, y_predicted))
    
    
    
def classify_mnist_reg():
    
    train = read_mnist('mnist_train.csv')
    valid = read_mnist('mnist_valid.csv')
    test = read_mnist('mnist_test.csv')
    # show_mnist('mnist_test.csv','pixels')
    
    # add a regularizer of your choice to classify_mnist()
    features_train, labels_train = data_preprocess_mnist(train)
    features_val, labels_val = data_preprocess_mnist(valid)
    features_test, labels_test = data_preprocess_mnist(test)
    
    # Convert to 0-1 scale to reduce computational cost
    features_train /= 255.0
    features_val /= 255.0
    features_test /= 255.0
    
    # Convert to PyTorch Tensor dataset
    train_dataset = TensorDataset(features_train, labels_train)
    val_dataset = TensorDataset(features_val, labels_val)
    test_dataset = TensorDataset(features_test, labels_test)
    
    # Non-lazily load in dataset so that we can split the data into batches, and shuffle the data
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Construct FFNN using the defined class. MNIST dataset has 784 pixels, and 10 digits (0-9)
    # Choose 128 hidden layers for more capacity
    model = FFNN_REG(784, 128, 10) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    num_epochs = 300  # Set the number of epochs
    y_predicted = []

    # create lists to store average losses
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            # Forward pass
            logits = model(X_batch)

            # Compute loss
            loss = criterion(logits, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                logits_val = model(X_val)
                loss_val = criterion(logits_val, y_val)
                total_val_loss += loss_val.item()


        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")
        
    # Evaluate on Test Set
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for X_test, y_test in test_loader:
            logits_test = model(X_test)
            loss_test = criterion(logits_test, y_test)
            total_test_loss += loss_test.item()

            prob = model.softmax(logits_test)
            # Make the prediction:
            predicted_class = torch.argmax(prob, dim=1)
            predicted_class = predicted_class.tolist()

            y_predicted.append(predicted_class)

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    print("======================================= Classification Finished =======================================")
    
    y_predicted = [item for sublist in y_predicted for item in sublist]
    print(classification_report(labels_test, y_predicted))
    
def classify_insurability_manual():
    
    train = read_insurability('three_train.csv')
    valid = read_insurability('three_valid.csv')
    test = read_insurability('three_test.csv')
    
    # reimplement classify_insurability() without using a PyTorch optimizer.
    # this part may be simpler without using a class for the FFNN
    
    
def main():
    classify_insurability()
    classify_mnist()
    classify_mnist_reg()
    #classify_insurability_manual()
    
if __name__ == "__main__":
    main()
