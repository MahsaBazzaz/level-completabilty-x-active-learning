import argparse
from datetime import datetime
import pdb
from classifier import Model
from logger import log
from passive_train import train_passive
from constants import CAVE_CHANNELS, CAVE_COLS, CAVE_ROWS, MARIO_CHANNELS, MARIO_COLS, MARIO_ROWS, SUPERCAT_CHANNELS, SUPERCAT_COLS, SUPERCAT_ROWS

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split

import numpy as np
import random

from modal.modAL.models.learners import ActiveLearner
from skorch import NeuralNetClassifier

import subprocess
from util import get_dataset, get_level
import json
from modal.modAL.uncertainty import uncertainty_sampling
from modal.modAL.uncertainty import margin_sampling
from modal.modAL.uncertainty import entropy_sampling

import warnings
warnings.filterwarnings("ignore")


def random_query_strategy(classifier, X, n_instances=1):
    # Generate a list of indices to select random instances from X
    indices = list(range(len(X)))
    # Shuffle the indices to select random instances
    random.shuffle(indices)
    # Return the first n_instances instances from the shuffled indices
    return indices[:n_instances]

def train_active(X_full, y_full, n_initial, strategy, n_queries, n_instances = 1):
    count = 0
    x_train, x_test, y_train, y_test = train_test_split(X_full, y_full, train_size=0.80)
    initial_idx = np.random.choice(range(len(x_train)), size=n_initial, replace=False)
    X_ini, y_ini = x_train[initial_idx].numpy(), y_train[initial_idx]
    # pdb.set_trace()
    y_ini = torch.tensor(y_ini, dtype=torch.float32).numpy()
    # cols, rows, channels
    # X_full[0].shape
    classifier = NeuralNetClassifier(Model(cols, rows, channels),
                                        criterion=nn.BCELoss,
                                        # criterion=nn.CrossEntropyLoss,
                                        optimizer=torch.optim.Adam,
                                        # optimizer__weight_decay=0.001,
                                        train_split=None,
                                        optimizer__lr = 0.0001,
                                        verbose=0,
                                        device=device,
                                        warm_start = True
                                 )
    log('classifier initiated', 'green')
    learner = ActiveLearner(estimator=classifier,X_training=X_ini, y_training=y_ini,
                             query_strategy=strategy)
    log('learner initiated', 'green')
    unqueried_score = learner.score(x_test, np.argmax(y_test, axis=1))
    log('initial score' + str(unqueried_score), 'green')
    performance_history = [unqueried_score]
    x_axis = []
    x_axis.append(count)
    save(performance_history, x_axis, count)

    # for idx in range(n_queries):
    while (abs(performance_history[-1] - (max_accuracy)) >= 0.01) and count<=200:

        query_index, query_instace = learner.query(x_train, n_instances = n_instances)
        X, y = x_train[query_index], y_train[query_index]
        # y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).numpy()
        y = torch.tensor(y, dtype=torch.float32).numpy()
        learner.teach(X=X.numpy(), y=y)
        count+= n_instances
        x_axis.append(count)
        x_train = np.delete(x_train, query_index, axis=0)
        y_train = np.delete(y_train, query_index, axis=0)

        model_accuracy = learner.score(x_test, np.argmax(y_test, axis=1))
        performance_history.append(model_accuracy)
        save(performance_history, x_axis, count)

    return (performance_history, x_axis)

def save(accuracy, x_axis, count, passive_accuracy=None):
    # Load existing data
    try:
        with open(folder + '/report_' + game + '.json', 'r') as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        data = []

    # Add new data to the existing array
    data.append({
        "accuracy": accuracy,
        "count": count,
        "x_axis": x_axis,
        "timestamp": datetime.now().strftime("%Y%m%d%H%M%S"),
        "active_accuracy" : passive_accuracy
    })

    # Save the updated array to the file
    with open(folder + '/report_' + game + '_' + idx + '.json', 'w') as json_file:
        json.dump(data, json_file, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Resnet classifier')

    parser.add_argument('--game', required=True, type=str)
    parser.add_argument('--idx', required=True, type=str)
    parser.add_argument('--criteria', required=True, type=str, help='random_query_strategy, margin_sampling, entropy_sampling, uncertainty_sampling')
    parser.add_argument('--n_ini', required=True, type=int)
    parser.add_argument('--n_instances', required=False, type=int)
    parser.add_argument('--n_query', required=False, type=int)

    args = parser.parse_args()
    game = args.game
    idx = args.idx
    folder = 'models'
    criteria = args.criteria
    n_ini = args.n_ini
    n_instances = args.n_instances
    n_query = args.n_query

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
    levels = np.transpose(levels, (0, 3, 1, 2))
    X_full = torch.tensor(levels, dtype=torch.float32)
    y_full = torch.tensor(labels,  dtype=torch.long)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        with open('models/metrics_' + game + '.json', 'r') as json_file:
            data = json.load(json_file)
            max_accuracy = data["accuracy"]
    except FileNotFoundError:
        print('No Passive history Found.')
        x_train, x_test, y_train, y_test = train_test_split(X_full, y_full, train_size=0.80)
        lr = 0.0001
        model = Model(cols, rows, channels)
        criterion = nn.BCELoss()
        # criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
        # optim.RMSprop(model.parameters(), lr=lr)
        num_epochs = 100
        batch_size = 32
        max_accuracy = train_passive(model, optimizer, criterion, num_epochs, x_train, y_train, x_test, y_test, batch_size, game, lr)

    if criteria == 'random':
        strategy = random_query_strategy
    elif criteria == 'margin':
        strategy = margin_sampling
    elif criteria == 'uncertainty':
        strategy = uncertainty_sampling
    elif criteria == 'entropy':
        strategy = entropy_sampling
    performance_history, x_axis = train_active(X_full, y_full, n_ini, strategy, n_query, n_instances)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_axis, performance_history)
    ax.scatter(x_axis, performance_history, s=13)
    ax.set_ylim(bottom=0, top=1)
    # ax.set_xticks([x for x, y in zip(x_axis, performance_history)])
    ax.axhline(y=(max_accuracy), color='r', linestyle='--', label='Passive Learner Accuracy with N=6000')
    ax.grid(True)
    ax.set_title('Incremental classification accuracy')
    ax.set_xlabel('Query iteration')
    ax.set_ylabel('Classification Accuracy')
    plt.legend()
    fig.savefig(folder + '/accuracy_' + game + '_' + idx + '.png')
