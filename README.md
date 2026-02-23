# ME5920 - Homework 1

This repository contains my solutions for **ME5920: Data Analytics and Machine Learning for Cyber-Physical Systems Applications**.

## Repository StructureME5920/
│── hw1/

│ ├── part2.ipynb # Problems 2.1, 2.2, 2.3

│ ├── part3.ipynb # Problems 3.1, 3.2


---

## Homework Overview (HW1)

### 🔹 Problem 2 - Programming & Exploratory Analytics

#### 2.1 Images (MNIST)
- Created a subset of 1000 samples using `torchvision`
- Plotted class distribution histogram
- Batched data using `einops` ([1000, 28, 28] → [Batches, 25, 1, 28, 28])
- Visualized pixel intensity using a 3D plot

#### 2.2 Time Series Analysis
- Plotted energy consumption over full duration and one-week period
- Generated hourly heatmap of energy usage
- Visualized distribution using histogram
- Created NSM (seconds from midnight) feature
- Analyzed relationships:
  - Energy vs NSM
  - Energy vs Pressure

#### 2.3 Multivariate Statistical Analysis
Computed descriptive statistics for airfoil dataset:
- Mean
- Median
- Standard Deviation / Variance
- Skewness
- Kurtosis
- Range

---

### Problem 3 - Image Processing

#### 3.1 Image Preprocessing & Whitening
- Generated augmented images (rotation, shift, scaling, warping)
- Extracted local patches from images
- Applied ZCA whitening to decorrelate features
- Compared channel-wise distributions:
  - Original images
  - Whitened images

#### 3.2 Image Segmentation (Soybean Subplots)
- Designed a fully automated pipeline (no hardcoding)
- Steps:
  - HSV thresholding to detect plant regions
  - ROI extraction using projection analysis
  - Grid-based division (6 × 6 structure)
  - Connected component analysis for bounding boxes
  - Numbered each subplot in consistent order
- Output:
  - Image with bounding boxes and subplot labels

---

## Requirements

Install dependencies:

```bash
pip install numpy pandas matplotlib opencv-python torchvision einops scipy
