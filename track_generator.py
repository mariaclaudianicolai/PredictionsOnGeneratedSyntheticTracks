import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def generate_tracks(NR, num_points, memory, sigma_noise):
    # Initialiaze x and y
    x = [0]
    y = [0]

    # Generation of the track
    for t in range(1, num_points):
        # Calculate directionality of the movement
        direction_x = x[-1] - x[-2] if len(x) > 1 else 0
        direction_y = y[-1] - y[-2] if len(y) > 1 else 0

        # If change direction or not
        if NR[t] < memory:
            new_x = x[-1] + direction_x
            new_y = y[-1] + direction_y
        else:
            new_x = x[-1] + np.random.normal(0, sigma_noise)
            new_y = y[-1] + np.random.normal(0, sigma_noise)

        # Add coordinates to lists
        x.append(new_x)
        y.append(new_y)
    return x, y


def plot_generated_track(x, y):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, linestyle='-', marker='o', markersize=3)
    plt.title('Cell track')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    # plt.show()
    plt.close()


def save_generated_coordinates(dict, directory):
    df_generated_tracks = []
    total_tracks = range(1000)
    for track_id in tqdm(total_tracks, desc='Processing IDs'):
        for i in range(len(dict['memory'])):
            x, y = generate_tracks(NR, num_points, dict['memory'][i], dict['sigma_noise'][i])
            df = pd.DataFrame(list(zip(x, y)), columns=['x', 'y'])
            df['track_id'] = track_id
            df['memory'] = dict['memory'][i]
            df['class'] = dict['class_name'][i]
            # Save the DataFrame to the list
            df_generated_tracks.append(df)
    # Concatenate all DataFrames in df_result list
    df_generated_tracks = pd.concat(df_generated_tracks)
    print(df_generated_tracks)
    # Save concatenated DataFrame to CSV
    # df_generated_tracks.to_csv(os.path.join(directory, 'generated_coordinates_{}.csv'.format(dict['memory'])), sep=',', index=False)
    return df_generated_tracks

def normalize_nparray(data: np.array, min_bound: float, max_bound: float, min_data: float, max_data: float) -> np.array:
    val_range = max_data - min_data
    tmp = (data - min_data) / val_range
    return tmp * (max_bound - min_bound) + min_bound

def normalize_coordinates(df, directory):
    df_norm = []
    for name, group in df.groupby(['track_id', 'class']):
        # normalize
        min_val = min(group['x'].min(), group['y'].min())
        max_val = max(group['x'].max(), group['y'].max())
        group['x_norm'] = normalize_nparray(group['x'], 0, 1, min_val, max_val)
        group['y_norm'] = normalize_nparray(group['y'], 0, 1, min_val, max_val)
        df_norm.append(group)
    df_norm = pd.concat(df_norm)
    df_norm.to_csv(os.path.join(directory, 'generated_coordinates_{}.csv'.format(dict['memory'])), sep=',', index=False)
    return df_norm


if __name__ == '__main__':
    base_dir = '/mnt/c/Users/Claudia/PycharmProjects/synthetic_tracks/classification'

    # number of points for the track
    NR = np.random.rand(100)
    num_points = 100
    linewidth = 1

    dict = {'memory': [0.9, 0.7], 'sigma_noise': [0.1, 0.3], 'class_name': ['a', 'b']}
    # dir_name = 'tracks_l{}_0907'.format(linewidth)

    directory = os.path.join(base_dir, 'datasets')
    os.makedirs(directory, exist_ok=True)

    # generate new tracks and save them in a csv
    # df_generated_tracks = save_generated_coordinates(dict, directory)

    # open reference df
    df_ref_tracks = pd.read_csv(os.path.join(directory, 'generated_coordinates_0907_ref.csv'))
    df_ref_tracks = df_ref_tracks.drop(['x_moved', 'y_moved'], axis=1)
    # normalize
    df_norm = normalize_coordinates(df_ref_tracks, directory)

    print('\nDone!')
