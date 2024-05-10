import os

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import pandas as pd


def count_prediction_errors(df: type(pd.DataFrame), dataset: str, exp_name: str):
    print('\ntotal tracks in {}_{}: {}'.format(dataset, exp_name, df.shape[0]))
    # Compare two columns for equality
    count_errors = (df['actual_class'] != df['predicted_class']).sum()
    return count_errors, df.shape[0]


def calculate_accuracy(df: type(pd.DataFrame)) -> float:
    correct_predictions = len(df[df['actual_class'] == df['predicted_class']])
    total_predictions = len(df)
    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy


def compute_confusion_matrix(df, dataset, exp_name):
    cf_mtx = confusion_matrix(df['actual_class'], df['predicted_class'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_mtx)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix\n{}_{}'.format(dataset, exp_name))
    plt.show()


def main():
    base_dir = '/mnt/c/Users/Claudia/PycharmProjects/SyntheticTracksGenerator/classification'

    dataset = 'aliased_tracks_112_l1'
    exp_name = 'exp_1'

    # Open prediction CSV
    prediction_results = os.path.join(base_dir, 'experiments', '{}_{}'.format(dataset, exp_name))
    df = pd.read_csv(os.path.join(prediction_results, 'test_predictions_{}.csv'.format(exp_name)), sep=',')

    # Find errors in predictions
    error_count, len_test_set = count_prediction_errors(df, dataset, exp_name)
    print('\tprediction errors: {}'.format(error_count))

    # Compute accuracy
    accuracy = calculate_accuracy(df)
    print('\tAccuracy for {}_{}: {:.2f}%'.format(dataset, exp_name, accuracy))

    compute_confusion_matrix(df, dataset, exp_name)


if __name__ == '__main__':
    main()

    print('\nDone!')
