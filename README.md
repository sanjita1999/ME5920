# ME5920 – Data Analytics and Machine Learning for Cyber-Physical Systems

This repository contains my coursework for **ME5920: Data Analytics and Machine Learning for Cyber-Physical Systems Applications** at **Iowa State University**.

**Student:** Sanjita Prajapati  

---

# Repository Structure

```
ME5920/
│
├── hw1/
│ ├── part2.ipynb
│ └── part3.ipynb
│
├── hw2/
│ ├── code_hw2.ipynb
│ ├── pca_reconstruction_outputs/
│ └── videos_for_Ag_and_Image/
│
├── hw3/
│ └── ucf11_video_classification_assignment.ipynb
│
└── README.md
```

---

# Homework 1 – Exploratory Analytics & Image Processing

Homework 1 focuses on **data exploration, statistical analysis, and image processing techniques**.

## Problem 2 – Programming & Exploratory Analytics

### 2.1 MNIST Image Analysis
- Created a subset of **1000 MNIST samples** using `torchvision`
- Plotted **class distribution histogram**
- Batched data using `einops`
- Visualized **pixel intensity in 3D**

### 2.2 Time Series Analysis
Energy consumption dataset analysis:

- Plotted **energy consumption over full duration**
- Generated **one-week time series visualization**
- Created **hourly heatmap of energy usage**
- Generated **NSM (seconds from midnight)** feature
- Analyzed relationships:
  - Energy vs NSM
  - Energy vs Pressure

### 2.3 Multivariate Statistical Analysis

Computed descriptive statistics for the **Airfoil dataset**:

- Mean
- Median
- Standard Deviation / Variance
- Skewness
- Kurtosis
- Range

---

## Problem 3 – Image Processing

### 3.1 Image Augmentation & Whitening

- Generated augmented images:
  - rotation
  - translation
  - scaling
  - warping
- Extracted **local image patches**
- Applied **ZCA whitening**
- Compared distributions between:
  - original images
  - whitened images

---

### 3.2 Soybean Subplot Segmentation

Designed a **fully automated pipeline (no hardcoding)** for detecting soybean subplots.

Pipeline steps:

- HSV thresholding to detect plant regions
- ROI extraction using projection analysis
- Grid-based segmentation (**6 × 6 structure**)
- Connected component analysis
- Automatic subplot numbering

Output:
- Image with **bounding boxes and subplot labels**

---

# Homework 2 – PCA-Based Video Analytics

Homework 2 explores **video analytics using Principal Component Analysis (PCA)** on a bird’s-eye American football video.

All implementations are contained in:

```
hw2/code_hw2.ipynb
```

---

## Methods Implemented

### 1. Low-Rank Video Modeling (PCA on Raw Frames)

- Converted video frames into matrix representation
- Applied PCA across the temporal dimension
- Computed cumulative explained variance
- Reconstructed frames using:
  - 1 component
  - 5 components
  - 20 components

---

### 2. Keyframe Extraction via Reconstruction Error

- PCA performed using **10 components**
- Reconstructed each frame
- Computed **L2 reconstruction error**
- Extracted frames with highest reconstruction error
- These correspond to **motion-heavy moments in the video**

---

### 3. Keyframe Extraction via PCA Projection Magnitude

- Projected frames into PCA space
- Computed **L2 magnitude of PCA coordinates**
- Selected frames farthest from PCA origin
- Visualized trajectory in **PC1–PC2 space**

---

### 4. PCA on Frame Differences (Motion Modeling)

Computed temporal frame differences:

```
D_t = F_t − F_(t−1)
```

Applied PCA to difference frames to isolate **motion patterns**.

This suppresses static background elements such as:

- football field
- grass texture
- yard lines

and highlights **moving players and coordinated motion**.

---

### 5. Low-Rank and Sparse Decomposition (Robust PCA Perspective)

Modeled the video as:

```
X = L + S
```

Where:

- **L** = low-rank component representing static background
- **S** = sparse component representing dynamic motion

Results show:

- low-rank component captures **field structure**
- sparse component highlights **moving players**

---

# Homework 3 - Video Classification using Deep Learning

Homework 3 focuses on **video classification using deep learning models** on the UCF11 dataset. The goal is to compare different approaches for modeling spatial and temporal information in videos.

All implementations are contained in:
hw3/ucf11_video_classification_assignment.ipynb


---

## Methods Implemented

### 1. Data Preparation

- Processed **1600 videos across 11 action classes**  
- Extracted:
  - video paths  
  - class labels  
  - group IDs  
- Created dataset splits:
  - Training: 1020 videos  
  - Validation: 173 videos  
  - Test: 407 videos  
- Used **folders 20–25 as the test split**

---

### 2. 2D CNN Classification (Frame-Based)

- Extracted **random frames** from each video  
- Resized frames to **112 × 112**  
- Trained a custom **2D CNN model**

Key experiment:
- Evaluated **multiple random frames per video**
- Averaged predictions across frames

Results:
- Best performance achieved with **5 frames per video**  
- Test accuracy: **~27%**

Insights:
- Performs well for **visually distinctive actions**
- Struggles with **motion-dependent classes**

---

### 3. 3D CNN Classification (Spatiotemporal Modeling)

- Sampled **clips of consecutive frames**  
- Trained a **3D CNN**

Results:
- Performance comparable to 2D CNN

Insights:
- Captures motion directly  
- Computationally expensive  
- Requires careful tuning  

---

### 4. CNN + LSTM (Temporal Modeling)

- Used pretrained **ResNet18** for feature extraction  
- Applied **LSTM** for temporal modeling  

Results:
- Achieved the **best performance among all models**

Insights:
- Captures temporal dependencies effectively  
- More efficient than 3D CNN  

---

## Key Takeaways

- **2D CNN:** Efficient but lacks temporal understanding  
- **3D CNN:** Captures motion but computationally heavy  
- **CNN + LSTM:** Best balance of performance and efficiency  

Overall, the results highlight the importance of **temporal modeling in video classification tasks**.

---

# Tools Used

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- OpenCV  
- Scikit-learn  
- Torchvision  
- Einops  
- SciPy  

---

# Installation

```bash
pip install numpy pandas matplotlib opencv-python scikit-learn torchvision einops scipy



---

# Tools Used

- Python
- NumPy
- Pandas
- Matplotlib
- OpenCV
- Scikit-learn
- Torchvision
- Einops
- SciPy

---

# Installation

Install dependencies using:

```bash
pip install numpy pandas matplotlib opencv-python scikit-learn torchvision einops scipy
```

---

# Running the Code

Homework 1 notebooks:

```
hw1/part2.ipynb
hw1/part3.ipynb
```

Homework 2 notebook:

```
hw2/code_hw2.ipynb
```

Homework 3 notebook:

hw3/ucf11_video_classification_assignment.ipynb 

Run all notebook cells sequentially to reproduce the analysis and visualizations.

---

# Notes

This repository contains coursework for **ME5920 – Data Analytics and Machine Learning for Cyber-Physical Systems Applications**.
