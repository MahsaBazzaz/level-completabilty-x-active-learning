         
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.interpolate import interp1d

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Resnet classifier')

    parser.add_argument('--game', required=True, type=str)

    parser.add_argument('--method', required=False, type=str)
    parser.add_argument('--average', required=False, type=bool, default=False)

    parser.add_argument('--folder', required=False, type=str, default="models")

    args = parser.parse_args()
    game = args.game
    method = args.method
    average = args.average
    folder = args.folder

    if average is False:
        try:
            with open(folder + '/report_' + game + '_' + method +  '.json', 'r') as json_file:
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
        fig.savefig(folder + '/accuracy_' + game + '_' + method + '.png')

    else:
        methods = ["random", "margin", "uncertainty", "entropy"]
        colors = ['r', 'g', 'b', 'c']
        # Define bin edges for x

        # Step 1: Calculate average accuracy for each fold
        for m, color in zip(methods, colors):
            with open(folder + '/report_' + game + '_' + m + '.json', 'r') as json_file:
                data = json.load(json_file)

                # Prepare dictionaries to store total x-axis values and counts for each accuracy
                total_x_axis = {}
                counts = {}

                # Calculate total x-axis values and counts for each accuracy across all folds
                for fold_data in data.values():
                    for entry in fold_data:
                        for idx, acc in enumerate(entry['accuracy']):
                            if acc not in total_x_axis:
                                total_x_axis[acc] = entry['x_axis'][idx]
                                counts[acc] = 1
                            else:
                                total_x_axis[acc] += entry['x_axis'][idx]
                                counts[acc] += 1

                # Calculate average x-axis for each accuracy
                avg_x_axis = {acc: total_x_axis[acc] / counts[acc] for acc in total_x_axis}

                # Sort the accuracies for plotting
                sorted_accs = sorted(avg_x_axis.keys())

                for avg_x, acc in zip(avg_x_axis.values(), sorted_accs):
                    plt.plot(avg_x, acc, color + 'o')
                # plt.plot(list(avg_x_axis.values()), sorted_accs, color + '-', label=m, alpha=0.5)
                plt.plot(0, 0, color + '-', label=m, alpha=0.5)

                # Create interpolation function
                # interp_func = interp1d(list(avg_x_axis.values()), sorted_accs, kind='nearest')  # kind='linear' for linear interpolation
                # x_new = np.linspace(min(list(avg_x_axis.values())), max(list(avg_x_axis.values())), 100)
                # y_new = interp_func(x_new)
                # plt.plot(x_new, y_new, color + '-', label=m, alpha=0.5)

                # swap
                # for avg_x, acc in zip(avg_x_axis.values(), sorted_accs):
                #     plt.plot(acc, avg_x, color + 'o')
                # plt.plot(sorted_accs, list(avg_x_axis.values()), color + '-', label=m, alpha=0.5)

# Plotting
plt.xlabel('Average Query Number between Folds')
plt.xlim(0, 60)
plt.ylabel('Accuracy')
plt.title(f'Average Query Number for Each Method in {game}')
plt.legend()
plt.grid(True)
plt.show()
