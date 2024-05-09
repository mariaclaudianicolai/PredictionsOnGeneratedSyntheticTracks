import os

import pandas as pd
import argparse
import numpy as np

import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import DataGenerator


def create_df(file_path):
    df = pd.DataFrame(columns=['id', 'actual_class', 'filename', 'filename_fullpath', 'prediction'])
    files = os.listdir(file_path)
    filenames = [filename for filename in files]
    ids = [filename.split('_')[1] for filename in files]
    classes = [filename.split('_')[-1][0] for filename in files]
    filenames_fullpath = [(file_path + '/' + filename) for filename in files]
    df['id'] = ids
    df['actual_class'] = classes
    df['filename'] = filenames
    df['filename_fullpath'] = filenames_fullpath
    print('\n{}:\nnum filenames: {}\nnum classes: {}'.format(file_path, len(df['filename']), len(df['actual_class'].unique())))
    return df


def build_model(input_shape, class_count, learning_rate):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    # model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(class_count, activation='sigmoid'))
    # model.add(Dense(class_count, activation='softmax'))

    model.summary()

    # Compile the model.
    model.compile(
        # loss='categorical_crossentropy',
        loss='binary_crossentropy',
        metrics=['accuracy'],
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    )
    return model


def plot_loss(path, training_loss, validation_loss, plot_name):
    plt.plot(training_loss, label='Training')
    plt.plot(validation_loss, label='Validation')
    plt.ylim(0, 0.8)
    plt.title('Loss plot')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(path, plot_name))
    plt.close()


def main():
    # Create command line argument parser
    parser = argparse.ArgumentParser()
    # Add command line options
    parser.add_argument("-d", "--dataset", type=str, help="Name of dataset to use in the experiment", required=True)
    parser.add_argument("-n", "--exp_name", type=str, help="Path output results (use same name of your dataset)", required=True)
    parser.add_argument("-i", "--image_size", type=int, help="Image size", default=112)
    parser.add_argument("-l", "--learning_rate", type=float, help="Learning rate", default=0.001)
    parser.add_argument("-e", "--epochs", type=int, help="Num of epochs", default=30)
    parser.add_argument("-b", "--base_dir", type=str, help="Base directory",
                        default='/mnt/c/Users/Claudia/PycharmProjects/SyntheticTracksGenerator/classification')
    # Get user options
    args = parser.parse_args()

    # directory to save outputs
    saving_outputs_path = str(os.path.join(args.base_dir, 'experiments', '{}_{}'.format(args.dataset, args.exp_name)))
    os.makedirs(saving_outputs_path, exist_ok=True)

    # Define image parameters
    image_height = args.image_size
    image_width = args.image_size
    # Define model parameters
    num_channels = 1
    input_shape = (image_height, image_width, num_channels)
    batch_size = 32
    classes = 2

    # Build the CNN model
    print('\nDefine model...')
    model = build_model(input_shape, classes, args.learning_rate)
    base_model_path = os.path.join(saving_outputs_path, 'base_model.keras')
    model.save_weights(base_model_path)

    # Create df with data from the images
    print('\nPreparing dataframe...')
    data_path = os.path.join(os.path.dirname(args.base_dir), 'datasets', args.dataset)

    data_df = create_df(data_path)

    class_counts = data_df['actual_class'].value_counts()  # descending order
    print('Class organization: {}'.format(class_counts))

    # Create train, validation and test dataframes
    # train_data_shuffled = data_df.sample(frac=1).reset_index(drop=True)
    split_percentage = round(data_df.shape[0] * 0.1)
    validation_data = data_df.iloc[0:split_percentage]
    test_data = data_df.iloc[split_percentage: split_percentage + split_percentage]
    train_data = data_df.iloc[split_percentage + split_percentage:]
    print('\nshape data_df: {}\nshape train: {}\nshape validation: {}\nshape test {}'.format(data_df.shape, train_data.shape, validation_data.shape,
                                                                                             test_data.shape))

    print('\n**Training**')
    # Load model
    model.load_weights(base_model_path)

    # Create data generators
    train_generator = DataGenerator.CustomSequence(train_data, input_shape, classes, batch_size, True)
    validation_generator = DataGenerator.CustomSequence(validation_data, input_shape, classes, batch_size, True)

    # Optional parameter for training
    # steps_per_epoch = np.ceil(train_generator.__len__() / 10)
    # validation_steps = np.ceil(validation_generator.__len__() / 2)

    history = model.fit(
        train_generator,
        epochs=args.epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=validation_generator,
        # steps_per_epoch=steps_per_epoch,
        # validation_steps=validation_steps
    )

    # Plot training and validation loss
    print(history.history.keys())
    plot_loss(saving_outputs_path, history.history['loss'], history.history['val_loss'], 'loss_plot.png')

    saved_model_name = os.path.join(saving_outputs_path, 'model.keras'.format(args.exp_name))
    model.save(saved_model_name)

    print('\n**Predicting**\n')
    # Load trained model
    model = load_model(saved_model_name)

    # Create test generator
    test_generator = DataGenerator.CustomSequence(test_data, input_shape, classes, 1, False)

    # Make prediction
    pred = model.predict(test_generator)

    predicted_classes = np.argmax(pred, axis=1)

    # Save prediction to CSV
    test_data.loc[:, 'prediction'] = predicted_classes
    test_data.loc[:, 'predicted_class'] = test_data['prediction'].replace([0, 1], ['a', 'b'])

    test_data.loc[:, 'prob_a'] = [p[0] for p in pred]
    test_data.loc[:, 'prob_b'] = [p[1] for p in pred]

    test_predictions_df_dir = os.path.join(saving_outputs_path, 'test_predictions_{}.csv'.format(args.exp_name))
    test_data.to_csv(test_predictions_df_dir, sep=',', index=False)

    print(
        'Experiment:\n\tdataset: {}\n\texp_name: {}\n\timage_size: {}\n\tepochs: {}\n\tlearning_rate: {}'.format(args.dataset, args.exp_name, args.image_size,
                                                                                                                  args.epochs,
                                                                                                                  args.learning_rate))


if __name__ == '__main__':
    print('Running...')

    main()

    print('\nDone!')
