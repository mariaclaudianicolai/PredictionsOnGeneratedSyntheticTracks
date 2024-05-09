import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_tracks(x: np.array, y: np.array, pixel: int, directory: str, output_name: str, linewidth: int):
    tracks_directory = os.path.join(directory, 'matplot_tracks')
    os.makedirs(tracks_directory, exist_ok=True)

    dpi = 80
    fig, ax = plt.subplots(figsize=(pixel / dpi, pixel / dpi), dpi=dpi)
    ax.plot(x, y, color='black', linewidth=linewidth)
    plt.grid(False)
    plt.axis('off')
    plt.savefig(os.path.join(tracks_directory, output_name))
    plt.close()


def main():
    base_dir = '/mnt/c/Users/Claudia/PycharmProjects/SyntheticTracksGenerator'
    datasets_dir = os.path.join(base_dir, 'datasets')

    # Open CSV with generated tracks
    tracks_file = os.path.join(datasets_dir, 'generated_tracks_0907.csv')
    df_generated_tracks = pd.read_csv(tracks_file, sep=',')

    # Plot tracks
    for name, group in df_generated_tracks.groupby(['track_id', 'class']):
        track_id, class_name = name
        x = group['x_norm'].values
        y = group['y_norm'].values
        plot_tracks(x, y, 112, datasets_dir, 'track_{}_{}_{}.png'.format(track_id, group['memory'].iloc[0], class_name), 1)


if __name__ == '__main__':
    print('Running...')

    main()

    print('\nDone!')
