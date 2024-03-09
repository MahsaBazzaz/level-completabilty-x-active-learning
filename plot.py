         
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt

import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Resnet classifier')

    parser.add_argument('--game', required=True, type=str)
    parser.add_argument('--idx', required=True, type=str)
    parser.add_argument('--folder', required=True, type=str)

    args = parser.parse_args()
    game = args.game
    idx = args.idx
    folder = args.folder

    try:
        with open(folder + '/report_' + game + '_' + idx +  '.json', 'r') as json_file:
            arr = json.load(json_file)
            performance_history = arr[-1]["accuracy"]
            x_axis = arr[-1]["x_axis"]
    except FileNotFoundError:
        print("ERROR")
        performance_history = []
        x_axis = []

    try:
        with open(folder + '/metrics_' + game + '.json', 'r') as json_file:
            data = json.load(json_file)
            max_accuracy = data["accuracy"]
    except FileNotFoundError:
        max_accuracy = 0
        print("ERROR")

    # Plot our performance over time.
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_axis, performance_history)
    ax.scatter(x_axis, performance_history, s=13)
    ax.set_ylim(bottom=0, top=1)
    # ax.set_xticks([x for x, y in zip(x_axis, performance_history)])
    ax.axhline(y=(max_accuracy * 0.01), color='r', linestyle='--', label='Passive Learner Accuracy with N=6000')
    ax.grid(True)
    ax.set_title('Incremental classification accuracy')
    ax.set_xlabel('Query iteration')
    ax.set_ylabel('Classification Accuracy')
    plt.legend()
    fig.savefig(folder + '/accuracy_' + game + '_' + idx + '.png')