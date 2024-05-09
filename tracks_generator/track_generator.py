import os
import string

import pandas as pd
import numpy as np
from tqdm import tqdm


def generate_tracks(track_points, num_generated_points, memory, sigma_noise):
    # Initialiaze x and y
    x = [0]
    y = [0]

    # Generation of the track
    for t in range(1, num_generated_points):
        # Calculate directionality of the movement
        direction_x = x[-1] - x[-2] if len(x) > 1 else 0
        direction_y = y[-1] - y[-2] if len(y) > 1 else 0

        # If change direction or not
        if track_points[t] < memory:
            new_x = x[-1] + direction_x
            new_y = y[-1] + direction_y
        else:
            new_x = x[-1] + np.random.normal(0, sigma_noise)
            new_y = y[-1] + np.random.normal(0, sigma_noise)

        # Add coordinates to lists
        x.append(new_x)
        y.append(new_y)
    return x, y


def create_df_generated_tracks(track_points, num_generated_points, total_tracks, memory):
    # Define empy list for df
    df_generated_tracks = []
    # Define a list of letters from 'a' to 'z'
    letters = list(string.ascii_lowercase)

    for track_id in tqdm(range(total_tracks), desc='Processing IDs'):
        # for i in range(len(memory)):
        for mem_idx, mem_value in enumerate(memory):
            sigma_noise = 1 - mem_value
            x, y = generate_tracks(track_points, num_generated_points, mem_value, sigma_noise)
            df = pd.DataFrame(list(zip(x, y)), columns=['x', 'y'])
            df['track_id'] = track_id
            df['memory'] = mem_value
            df['class'] = letters[mem_idx]
            # Save the DataFrame to the list
            df_generated_tracks.append(df)
    # Concatenate all DataFrames in df_result list
    df_generated_tracks = pd.concat(df_generated_tracks)
    # print(df_generated_tracks)
    return df_generated_tracks


def main():
    base_dir = '/mnt/c/Users/Claudia/PycharmProjects/SyntheticTracksGenerator'

    # Define number of points for track
    num_generated_points = 100
    # Generate random numbers
    track_points = np.random.rand(num_generated_points)
    # Define number of tracks to generate
    total_tracks = 1000
    # Define memories
    memories = [0.9, 0.7]

    directory = os.path.join(base_dir, 'datasets')
    os.makedirs(directory, exist_ok=True)

    # generate new tracks and save them in a csv
    df_generated_tracks = create_df_generated_tracks(track_points, num_generated_points, total_tracks, memories)
    print(df_generated_tracks)
    # Save concatenated DataFrame to CSV
    str_memories = ''.join(['{:02d}'.format(int(memory * 10)) for memory in memories])
    df_generated_tracks.to_csv(os.path.join(directory, 'generated_tracks_{}.csv'.format(str_memories)), sep=',', index=False)


if __name__ == '__main__':
    main()

    print('\nDone!')
