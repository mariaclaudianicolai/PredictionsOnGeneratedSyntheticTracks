import os
import string

import pandas as pd
import numpy as np
from tqdm import tqdm


def normalize_nparray(data: np.array, min_bound: float, max_bound: float, min_data: float, max_data: float) -> np.array:
    val_range = max_data - min_data
    tmp = (data - min_data) / val_range
    return tmp * (max_bound - min_bound) + min_bound


def normalize_coordinates(df: type(pd.DataFrame)) -> type(pd.DataFrame):
    df_norm = []
    for name, group in df.groupby(['track_id', 'class']):
        # Find min and max
        min_val = min(group['x'].min(), group['y'].min())
        max_val = max(group['x'].max(), group['y'].max())
        # Normalize [0, 1]
        group['x_norm'] = normalize_nparray(group['x'], 0, 1, min_val, max_val)
        group['y_norm'] = normalize_nparray(group['y'], 0, 1, min_val, max_val)
        df_norm.append(group)
    df_norm = pd.concat(df_norm)
    return df_norm


def main():
    base_dir = '/mnt/c/Users/Claudia/PycharmProjects/SyntheticTracksGenerator'

    # Open CSV with generated tracks
    tracks_file = os.path.join(base_dir, 'datasets', 'generated_tracks_0907.csv')
    df_generated_tracks = pd.read_csv(tracks_file, sep=',')

    # Normalization [0, 1]
    df_normalized_tracks = normalize_coordinates(df_generated_tracks)
    # Save results to CSV
    df_normalized_tracks.to_csv(tracks_file, sep=',', index=False)


if __name__ == '__main__':
    print('Running...')

    main()

    print('\nDone!')
