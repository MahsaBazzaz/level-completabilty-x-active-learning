         
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import json

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
        avg_accuracies = []
        x_axis_combined = None

        # Step 1: Calculate average accuracy for each fold
        for m in methods:
                with open(folder + '/report_' + game + '_' + m +  '.json', 'r') as json_file:
                    data = json.load(json_file)
                    # y.append(data[-1]["accuracy"]) 
                    # x.append(data[-1]["x_axis"])
                    for fold_data in data.values():
                        accuracy = fold_data[0]["accuracy"]
                        count = fold_data[0]["count"]
                        x_axis = fold_data[0]["x_axis"]
                        avg_accuracy = [sum(acc) / count for acc in zip(*[accuracy])]
                        avg_accuracies.append(avg_accuracy)
                        if x_axis_combined is None:
                            x_axis_combined = np.array(x_axis)
                        else:
                            x_axis_combined = np.maximum(x_axis_combined, np.array(x_axis))

                    # Step 2: Combine the average accuracies of all folds
                    combined_avg_accuracy = np.zeros(len(x_axis_combined))
                    for avg_accuracy in avg_accuracies:
                        combined_avg_accuracy[:len(avg_accuracy)] += np.array(avg_accuracy)

                    combined_avg_accuracy /= len(avg_accuracies)

                    # Step 3: Plot the combined average accuracies against the cumulative number of queries
                    plt.plot(x_axis_combined, combined_avg_accuracy, marker='o')
                    plt.xlabel('Number of Queries')
                    plt.ylabel('Average Accuracy')
                    plt.title('Average Accuracy vs Number of Queries')
                    plt.grid(True)
                    plt.show()