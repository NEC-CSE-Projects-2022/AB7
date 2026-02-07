Dataset Title

ShanghaiTech Part-A Crowd Counting Dataset

Usage of Dataset

The ShanghaiTech Part-A dataset is used to train and evaluate deep learning models for crowd counting and density estimation. It enables the model to learn how to estimate the number of people in highly congested scenes by generating density maps from annotated head locations. This dataset is particularly suitable for dense crowd analysis where traditional object detection methods fail due to heavy occlusion and scale variations.

Dataset Information

Dataset Name:
ShanghaiTech Part-A Crowd Counting Dataset

Source:
Shanghai Jiao Tong University

Domain:
Computer Vision, Image Processing

Task:
Crowd Counting via Density Map Estimation

Problem Type:
Supervised Regression

File Format:

Images: .jpg

Annotations: .mat (MATLAB files containing head coordinates)

Dataset Link:
https://www.kaggle.com/datasets/tthien/shanghaitech

Dataset Overview

Total Records:
482 images

Labeled Records:
482 (fully annotated)

Classes:
Single class (Person / Head)

Annotation Type:
Point-level head annotations converted into density maps

Why This Dataset?

Designed specifically for highly dense crowd scenes

Contains large variations in scale, perspective, and occlusion

Widely used benchmark in crowd counting research

Enables fair comparison with state-of-the-art methods

Ideal for evaluating density map-based models like EfficientSINet-B4

Features Used

Feature 1:
RGB image pixel intensities (visual information)

Feature 2:
Spatial distribution of head annotations (converted to density maps)

Feature 3:
Multi-scale visual features extracted using EfficientNet-B4

Summary

The ShanghaiTech Part-A dataset provides high-quality, densely populated crowd images with precise annotations, making it a standard benchmark for crowd counting research. Its challenging nature helps in developing robust and scalable models such as EfficientSINet-B4, which can accurately estimate crowd density in real-world scenarios.
