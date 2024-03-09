import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim

class Model(nn.Module):
    def __init__(self, cols, rows, channels):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(cols * rows * channels, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten()
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
        return output.item()
        # probability = output.item()
        # complement_probability = 1 - probability
        # return probability, complement_probability

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            _, predicted_class = torch.max(output, 1)
            return predicted_class


# class Model(nn.Module):
#     def __init__(self, input_shape, num_classes = 2):
#         super(Model, self).__init__()

#         # Stage 1
#         self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=3, padding=1)
#         self.batch_norm1 = nn.BatchNorm2d(64)
#         self.relu1 = nn.ReLU()
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

#         # Stage 2
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.batch_norm2 = nn.BatchNorm2d(64)
#         self.relu2 = nn.ReLU()
#         self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

#         # Stage 3
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.batch_norm3 = nn.BatchNorm2d(128)
#         self.relu3 = nn.ReLU()
#         self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

#         # Flatten and fully connected layers
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(128 * (input_shape[1] // 8) * (input_shape[2] // 8), 256)
#         self.relu4 = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(256, num_classes)
#         # self.fc2 = nn.Linear(256, num_classes, nn.Sigmoid())

#     def forward(self, x):
#         # Stage 1
#         x = self.conv1(x)
#         x = self.batch_norm1(x)
#         x = self.relu1(x)
#         x = self.maxpool1(x)

#         # Stage 2
#         x = self.conv2(x)
#         x = self.batch_norm2(x)
#         x = self.relu2(x)
#         x = self.maxpool2(x)

#         # Stage 3
#         x = self.conv3(x)
#         x = self.batch_norm3(x)
#         x = self.relu3(x)
#         x = self.maxpool3(x)

#         # Flatten and fully connected layers
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.relu4(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         # x = self.fc2(x).sigmoid()

#         return x