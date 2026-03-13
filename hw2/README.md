# Homework 2 – PCA-Based Video Analytics

**Course:** ME5920 – Image and Video Analytics  
**Student:** Sanjita Prajapati

---

## Overview

This homework explores several **PCA-based techniques for video analysis** using a bird’s-eye American football video. The objective is to understand how dimensionality reduction techniques such as **Principal Component Analysis (PCA)** can be applied to analyze video structure, detect keyframes, and isolate motion patterns.

All implementations and results are contained in the notebook:

`code_hw2.ipynb`

The notebook includes the **complete code, visualizations, and explanations** for all five required parts of the assignment.

---

## Dataset

The dataset contains two videos:

- `football_video.mp4` – Bird’s-eye American football play  
- `soybean_timelapse.mp4` – Plant growth timelapse video  

For this assignment, the **football video** was used for analysis.

---

## Methods Implemented

### 1. Low-Rank Video Modeling (PCA on Raw Frames)

Each video frame is flattened and stacked into a matrix:

`X ∈ R^(T × HWC)`

Where:

- **T** = number of frames  
- **H, W, C** = frame height, width, and color channels  

PCA is applied to this matrix to identify dominant directions of variance across frames.

**Results include:**

- Cumulative explained variance plots
- Number of components required to explain **80%, 90%, and 95% variance**
- Video reconstruction using **1, 5, and 20 principal components**

This experiment demonstrates how PCA captures the **dominant structure of the scene**, while gradually losing fine visual details as fewer components are retained.

---

### 2. Keyframe Extraction via Reconstruction Error

PCA is performed using a fixed number of components (**k = 10**).  
Each frame is reconstructed from the reduced representation and the **L2 reconstruction error** is computed.

Frames with higher reconstruction error correspond to:

- motion-heavy moments
- complex formations
- event-rich scenes

The top 10 frames with the highest reconstruction error are extracted as **keyframes**.

---

### 3. Keyframe Extraction via PCA Projection Magnitude

Each frame is projected into PCA space and the **L2 magnitude of the PCA coordinates** is computed.

Frames that lie far from the origin in PCA space represent **visually distinctive scene configurations**.

This method highlights frames corresponding to:

- distinctive formations
- structural changes in the play
- unusual scene layouts

The trajectory of frames in **PC1–PC2 space** is also visualized to show how the video evolves over time.

---

### 4. PCA on Frame Differences (Motion Modeling)

To isolate motion information, temporal frame differences are computed:

`D_t = F_t − F_(t−1)`

PCA is applied to these difference frames.

This approach suppresses static background elements such as:

- football field
- grass texture
- yard lines

and highlights **moving players and motion patterns**.

Frames with high motion magnitude correspond to **periods of strong player movement**.

---

### 5. Low-Rank and Sparse Decomposition (Robust PCA Perspective)

The video is modeled as:

`X = L + S`

Where:

- **L** = low-rank component representing the static background
- **S** = sparse component representing dynamic foreground motion

The low-rank approximation is obtained using PCA reconstruction, and the sparse component is computed as the residual between the original frames and the low-rank reconstruction.

**Observations:**

- The **low-rank component** captures the football field and overall scene layout.
- The **sparse component** highlights moving players and dynamic motion regions.
- Frames with the largest sparse magnitude correspond to **event-rich moments** in the play.

---

## Key Observations

- PCA effectively models the **dominant visual structure** of the video.
- Reconstruction error highlights **motion-heavy frames**.
- PCA projection magnitude identifies **distinctive scene configurations**.
- Frame-difference PCA isolates **temporal motion patterns**.
- Low-rank + sparse decomposition separates **background and dynamic foreground motion**.

Together, these methods demonstrate how PCA can be used for **video compression, motion analysis, and event detection**.

---

## Repository Structure
