# Calibrating Neural Networks via Radius Regularization
- **Improving Neural Network Calibration with Radius-Based Regularization**
- **Bachelor Thesis at Sapienza University of Rome**
- **Research Area**: Model Calibration, Hyperbolic Geometry, Uncertainty Estimation 

## Overview
This project introduces a novel **radius-based regularization technique** for improving the calibration of neural networks. Inspired by **hyperbolic geometry**, the approach leverages **radius alignment** to enhance the reliability of model predictions. Our method significantly reduces **Expected Calibration Error (ECE)**, improving uncertainty estimation without requiring post-training calibration steps.
- **Developed for both Euclidean & Hyperbolic neural networks**
- **Reduces calibration errors (ECE, MCE, RMSCE) by up to 50%**
- **Outperforms traditional regularization methods (Label Smoothing, Focal Loss)**

## Paper
- **Title**: _Calibrating Neural Networks via Radius Regularization_
- **Status**: Under Review

## How It Works
1. **Radius-Based Confidence Calibration**: Predictive confidence is regularized using hyperbolic radii, ensuring better alignment with probability estimates.
2. **Hyperbolic & Euclidean Support**: Works on both hyperbolic and standard deep learning architectures.
3. **No Post-Processing Required**: Unlike traditional calibration techniques, our method **integrates into training** without additional tuning.

![Architecture](https://github.com/user-attachments/assets/65f3249e-f18c-4d39-b5e0-a4ca776f58ba)

## Future Work
- Extending radius-based regularization to **generative models**
