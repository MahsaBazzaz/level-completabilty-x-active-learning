
import pdb
from classifier import Model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from util import get_dataset
import argparse
import numpy as np
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from constants import CAVE_CHANNELS, CAVE_COLS, CAVE_ROWS, MARIO_CHANNELS, MARIO_COLS, MARIO_ROWS, SUPERCAT_CHANNELS, SUPERCAT_COLS, SUPERCAT_ROWS

def train(model, train_dataset, train_loader, val_dataset, val_loader, criterion, optimizer, num_epochs, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    # Train the model for the specified number of epochs
    for epoch in range(num_epochs):
        print("epoch: ", epoch)
        # Set the model to train mode
        model.train()

        # Initialize the running loss and accuracy
        running_loss = 0.0
        running_corrects = 0

        # Iterate over the batches of the train loader
        for inputs, labels in train_loader:
            # Move the inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the optimizer gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            predicted_labels = torch.argmax(outputs, dim=1)
            # Convert predicted labels to original label format
            preds = []
            for label in predicted_labels:
                if label == 0:
                    preds.append([1, 0])
                else:
                    preds.append([0, 1])
            loss = criterion(outputs, labels)

            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()

            # Update the running loss and accuracy
            running_loss += loss.item() * inputs.size(0)

            for result, label in zip(torch.tensor(preds), labels.data):
                if torch.all(torch.eq(result, label)):
                    running_corrects += 1

        # Calculate the train loss and accuracy
        train_loss = running_loss / len(train_dataset)
        train_acc = running_corrects / len(train_dataset)

        # Set the model to evaluation mode
        model.eval()

        # Initialize the running loss and accuracy
        running_loss = 0.0
        running_corrects = 0

        # Iterate over the batches of the validation loader
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Move the inputs and labels to the device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                predicted_labels = torch.argmax(outputs, dim=1)
                # Convert predicted labels to original label format
                preds = []
                for label in predicted_labels:
                    if label == 0:
                        preds.append([1, 0])
                    else:
                        preds.append([0, 1])
                loss = criterion(outputs, labels)

                # Update the running loss and accuracy
                running_loss += loss.item() * inputs.size(0)
                for result, label in zip(torch.tensor(preds), labels.data):
                    if torch.all(torch.eq(result, label)):
                        running_corrects += 1

        # Calculate the validation loss and accuracy
        val_loss = running_loss / len(val_dataset)
        val_acc = running_corrects / len(val_dataset)

        # Print the epoch results
        print('Epoch [{}/{}], train loss: {:.4f}, train acc: {:.4f}, val loss: {:.4f}, val acc: {:.4f}'
              .format(epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))
    
    return val_acc

def train_passive(game, model, weight_decay, learning_rate, num_epochs, x_train, y_train, x_test, y_test, batch_size):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(x_test, y_test)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Fine-tune the last layer for a few epochs
    train(model, train_dataset, train_loader, val_dataset, val_loader, criterion, optimizer, num_epochs)

    # Unfreeze all the layers and fine-tune the entire network for a few more epochs
    for param in model.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    accuracy = train(model, train_dataset, train_loader, val_dataset, val_loader, criterion, optimizer, num_epochs) 
    model_name = "./models/" + game + "_" + 'generic_classifier_py' + ".pth"
    torch.save(model.state_dict(), model_name)
    with open('models/metrics_' + game + '.json', "w") as json_file:
        json.dump({
                    "accuracy": accuracy * 100,
                    "num_epochs" : num_epochs,
                    "learning_rate" : learning_rate,
                    "weight_decay" : weight_decay,
                    "batch_size" : batch_size,
                    "criterion" : "CrossEntropyLoss",
                    "optimizer" : "Adam",
                    "timestamp": datetime.now().strftime("%Y%m%d%H%M%S")
                    }, json_file)      
    return accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Resnet classifier')

    parser.add_argument('--game', required=True, type=str, help='cave/icarus/mario')

    args = parser.parse_args()
    game = args.game
    if game == "cave":
        cols = CAVE_COLS
        rows = CAVE_ROWS
        channels = CAVE_CHANNELS
    elif game == 'mario':
        cols = MARIO_COLS
        rows = MARIO_ROWS
        channels = MARIO_CHANNELS
    elif game == 'supercat':
        cols = SUPERCAT_COLS
        rows = SUPERCAT_ROWS
        channels = SUPERCAT_CHANNELS

    levels, labels = get_dataset(game)
    x_train, x_test, y_train, y_test = train_test_split(levels, labels, train_size=0.80)
    # Define the optimizer and the learning rate
    lr = 0.0001
    model = Model(cols, rows, channels)
    # criterion = nn.BCELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    num_epochs = 100
    batch_size = 32
    accuracy = train_passive(model, optimizer, criterion, num_epochs, x_train, y_train, x_test, y_test, batch_size, game, lr)

