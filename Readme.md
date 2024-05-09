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
We'll make a classification by using CNN model.
