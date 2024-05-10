# Synthetic Track Prediction

## 1. Generating and Plotting Synthetic Tracks

We'll begin by generating coordinates (x and y) to create synthetic track lines. Specifically, we'll generate two classes of tracks.

### 1.1 Memory Concept

Each generated track shares a memory factor. This memory factor indicates the degree of similarity between tracks. For instance, tracks with a memory factor of
0.9 exhibit 90% similarity.

### 1.2 Normalization

We'll apply a normalization formula to ensure that the generated tracks fall within the range of 0 to 1.

### 1.3 Matplotlib plot

We'll plot tracks with Matplotlib.

### 1.4 Opencv plot

- We will use OpenCV to plot tracks, both with aliasing and antialiasing.
- The tracks will be shifted to the center of the image before plotting.

## 2. Classification

In this phase, we'll utilize a CNN model for classification.

### 2.1 Training

For training, the following options are available for the model:

```
-d <Name of dataset to use in the experiment>
-n <Give name to experiment>
-i <Image size> [default=112]
-l <Learning rate> [default=0.001]
-e <Num of epochs> [default=30]
-b <Base directory> [CHANGE WITH DIRECTORY OF SYSTEM]
```

Example usage: ```-d aliased_tracks_112_l1 -n exp_1 -e 20```

### 2.2 Evaluation

In the evaluation phase, we will:

- Count errors in prediction and calculate the accuracy.
- Calculate the confusion matrix.

Before running evaluation, ensure to set the **dataset** and **exp_name** according to the experiment you want to evaluate:

```
dataset = 'aliased_tracks_112_l1'
exp_name = 'exp_1'
```