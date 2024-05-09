import enum
import os

import numpy as np
import pandas as pd
import cv2


class LineType(enum.Enum):
    ALIASED = 'aliased'
    ANTIALIASED = 'antialised'


def write_tracks(datasets_dir, line_type, image_size, line_thickness: int, mtx: np.array, x: np.array, y: np.array, track_id: str, memory: str, class_name):
    directory = os.path.join(datasets_dir, '{}_tracks_{}_l{}'.format(line_type.value, image_size, line_thickness))
    os.makedirs(directory, exist_ok=True)

    for i in range(len(x) - 1):
        if line_type == LineType.ALIASED:
            cv2.line(mtx, (x[i], y[i]), (x[i + 1], y[i + 1]), (0, 0, 0), line_thickness, cv2.LINE_AA)
        elif line_type == LineType.ANTIALIASED:
            cv2.line(mtx, (x[i], y[i]), (x[i + 1], y[i + 1]), (0, 0, 0), line_thickness, cv2.LINE_8)
    mtx = cv2.flip(mtx, 0)
    cv2.imwrite(os.path.join(directory, 'track_{}_{}_{}.png'.format(track_id, memory, class_name)), mtx)


def normalize_nparray(data: np.array, min_bound: float, max_bound: float, min_data: float, max_data: float) -> np.array:
    val_range = max_data - min_data
    tmp = (data - min_data) / val_range
    return tmp * (max_bound - min_bound) + min_bound


def main():
    base_dir = '/mnt/c/Users/Claudia/PycharmProjects/SyntheticTracksGenerator'
    datasets_dir = os.path.join(base_dir, 'datasets')

    # Open CSV with generated tracks
    tracks_file = os.path.join(datasets_dir, 'generated_tracks_0907.csv')
    df_generated_tracks = pd.read_csv(tracks_file, sep=',')

    # Plot parameters
    img_size = 112
    line_thickness = 1

    # Plot tracks
    for name, group in df_generated_tracks.groupby(['track_id', 'class']):
        track_id, class_name = name
        # Create empty matrix
        mtx = np.ones((img_size, img_size), dtype=np.uint8) * 255
        # Find min and max
        min_data = min(group['x_norm'].min(), group['y_norm'].min())
        max_data = max(group['x_norm'].max(), group['y_norm'].max())
        # Scale to image size
        x = normalize_nparray(group['x_norm'].to_numpy(), 0, img_size, min_data, max_data).astype(np.int32)
        y = normalize_nparray(group['y_norm'].to_numpy(), 0, img_size, min_data, max_data).astype(np.int32)
        # Write aliased and antialiased tracks
        write_tracks(datasets_dir, LineType.ANTIALIASED, img_size, line_thickness, mtx, x, y, track_id, group['memory'].iloc[0], class_name)
        write_tracks(datasets_dir, LineType.ALIASED, img_size, line_thickness, mtx, x, y, track_id, group['memory'].iloc[0], class_name)


if __name__ == '__main__':
    print('Running...')

    main()

    print('\nDone!')
