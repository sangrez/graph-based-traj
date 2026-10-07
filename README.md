# Mobility Pattern Mining via Heterogeneous Graph Neural Networks

**Author:** [Your Name]  
**Date:** March 2026

## 1. Abstract / Introduction
This project presents a novel framework for analyzing urban mobility patterns using **Heterogeneous Graph Neural Networks (HeteroGNNs)**. Unlike traditional sequence-based models (RNNS, LSTMs) that treat trajectories as simple chains of coordinates, our approach models the entire urban mobility ecosystem as a complex network. By representing both **Geographic Regions** (H3 hexagons) and **User Trajectories** as distinct node types in a shared graph, we can learn rich, semantic embeddings that capture:
1.  **Transport Modes** (Walking vs. Driving vs. Public Transit)
2.  **Unlabelled Life Patterns** (Commuting, Leisure, Delivery behaviors)

This repository contains the complete implementation, from raw GPS processing to the generation of publication-ready visualization artifacts.

## 2. Methodology

### A. Spatial Discretization (H3 Hexagonal Grid)
Raw GPS data is noisy and continuous. We first discretize the city of Beijing into a hexagonal grid using Uber's **H3 system** (Resolution 9, approx. $0.1 \text{km}^2$ per cell). This transforms a trajectory $T = \{(lat_1, lon_1), \dots, (lat_n, lon_n)\}$ into a sequence of tokens $H = \{h_1, h_2, \dots, h_m\}$.

### B. Heterogeneous Graph Construction
We construct a graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with two types of nodes:
*   **Region Nodes ($V_R$):** represent physical location hexes.
    *   *Features:* Centroid Lat/Lon (Normalized).
*   **Trajectory Nodes ($V_T$):** represent individual trips.
    *   *Features:* Average Velocity ($m/s$), Log-Duration ($log(t)$), Displacement.

We define three types of edges to model interactions:
1.  **Region-Region ($E_{RR}$):** Spatial adjacency (Grid neighbors).
2.  **Trajectory-Region ($E_{TR}$):** "Visits" (A trajectory passes through a region).
3.  **Region-Trajectory ($E_{RT}$):** "Hosted by" (Reverse edge for message passing).

### C. Graph Neural Network Architecture
We employ a **Heterogeneous GraphSAGE** architecture. The model learns by aggregating messages across the different relation types:

$$
h_v^{(l+1)} = \sigma \left( \sum_{r \in \mathcal{R}} \sum_{u \in \mathcal{N}_r(v)} W_r^{(l)} \cdot h_u^{(l)} \right)
$$

Where $\mathcal{R}$ represents the relation types (e.g., *Visits*, *Neighbors*).
*   **Layer 1:** Regions aggregate info from neighbors (map context) and visiting trajectories (traffic flow).
*   **Layer 2:** Trajectories aggregate refined info from the regions they visited (path context).

### D. Learning Objectives
1.  **Supervised classification:** We train the model to classify transport modes (Car, Bus, Walk, etc.) on a labeled subset of the GeoLife data.
2.  **Unsupervised Clustering:** We project the learned embeddings of *unlabeled* trajectories into a latent space and perform K-Means clustering to discover hidden mobility patterns ("Life Patterns").

## 3. Dataset
*   **Source:** Microsoft GeoLife GPS Trajectories (Beijing subset).
*   **Scale:** ~18,000 trajectories processed.
*   **Classes:** 10 modes including *Walk, Bike, Bus, Car, Subway, Train*.
*   **Preprocessing:** Trajectories split by 20-minute gaps; filtered for valid GPS signals.

## 4. Experimental Results & Artifacts

The pipeline generates three key figures suitable for an IEEE workshop paper:

### Figure 1: Classification Performance (`Slide1_ConfusionMatrix.png`)
*   **Description:** A confusion matrix showing the model's ability to distinguish between transport modes.
*   **Analysis:** High accuracy on *Walk* vs. *Car* confirms the efficacy of Velocity features. Confusion between *Bus* and *Car* highlights the complexity of shared road networks.

### Figure 2: Physics-Aware Embeddings (`Slide2_Speed_Analysis.png`)
*   **Description:** Boxplots of Average Speed vs. Transport Mode.
*   **Analysis:** This validates the feature engineering. The clear separation (Walk < Bike < Bus < Car < Train) provides a strong signal for the GNN to learn from.

### Figure 3: Discovery of Life Patterns (`Slide3_Life_Patterns.html`)
*   **Description:** An interactive map visualizing the spatial distribution of 3 unsupervised clusters found in the unlabeled data.
*   **Interpretation:**
    *   *Cluster 0 (Red):* Short, dense trips in city center $\rightarrow$ likely **Last-mile delivery / Commute**.
    *   *Cluster 1 (Green):* Long, arterial trips $\rightarrow$ likely **Highway transit**.
    *   *Cluster 2 (Blue):* Parks and residential areas $\rightarrow$ likely **Leisure walking**.

## 5. How to Reproduce

### Prerequisites
*   Python 3.8+
*   PyTorch, PyTorch Geometric, Pandas, H3, Folium, Seaborn

### Steps
1.  **Preprocess Data:**
    ```bash
    python 01_preprocess.py
    ```
    *Extracts speed features and maps GPS to H3.*

2.  **Build Graph:**
    ```bash
    python 02_build_graph.py
    ```
    *Constructs the `geolife_graph.pt` HeteroData object.*

3.  **Train & Visualize:**
    ```bash
    python 03_train_and_evaluate.py
    ```
    *Trains the Model for 60 epochs and generates all Slide images.*

4.  **Similarity Search:**
    ```bash
    python 04_similarity_search.py
    ```
    *Demonstrates vector search for finding similar trips.*

## 6. Citation
If used, please cite the GeoLife dataset:
*Yu Zheng, Lizhu Zhang, Xing Xie, Wei-Ying Ma. Mining interesting locations and travel sequences from GPS trajectories. WWW 2009.*
