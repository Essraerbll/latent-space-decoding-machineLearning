# PCA-Based Dimensionality Reduction and Neural Network Reconstruction on MNIST

---

## 📚 Project Overview

This project investigates the effectiveness of **Principal Component Analysis (PCA)** for dimensionality reduction, followed by **Neural Network-based reconstruction** on the **MNIST** handwritten digits dataset.

The core idea is to compress high-dimensional image data (28x28 pixels = 784 features) into a low-dimensional latent space (3 components), and then reconstruct the original images using a **Multi-Layer Perceptron (MLPRegressor)** model.

Through this approach, the project demonstrates how well a simple neural network can recover detailed information from a highly compressed representation.

---

## 🛠 Methods and Tools

- **Data Source:**  
  - MNIST dataset (70,000 handwritten digit images) obtained via `fetch_openml`.

- **Dimensionality Reduction:**  
  - **Principal Component Analysis (PCA)** was applied to reduce the dimensionality from 784 features to 3 principal components.

- **Reconstruction Model:**  
  - **Multi-Layer Perceptron Regressor (MLPRegressor)** models were trained to reconstruct the original 784-dimensional data from the compressed 3-dimensional representations.
  - Training iterations were varied (10, 25, 500) to observe the impact on reconstruction quality.

- **Data Preprocessing:**  
  - Feature scaling using **StandardScaler** for optimal PCA performance.

- **Visualization:**  
  - Original images and reconstructed images were plotted side-by-side for direct qualitative comparison across different training stages.

---

## 📦 File Structure

| File Name    | Purpose                                           |
|--------------|---------------------------------------------------|
| `Task3.py`   | Main script for PCA, MLP training, and visualization |

No external data files are required; the MNIST dataset is automatically downloaded via OpenML.

---

## 🚀 How to Run

1. Install required libraries:
```bash
pip install numpy pandas scikit-learn matplotlib
```

2. Execute the Python script:
```bash
python Task3.py
```

Upon execution:
- A sample of 1,000 images will be processed.
- PCA transformation and MLP training will be performed.
- Reconstructed images will be visualized alongside their originals.

---

## 📈 Key Findings

- Even with only **3 principal components**, the MLPRegressor was able to reconstruct recognizable versions of the original MNIST images.
- Reconstruction quality improves significantly as training iterations increase (10 ➔ 25 ➔ 500).
- Early iterations produce blurry reconstructions, while extended training captures more detailed structures.

---

## ✨ Motivation

High-dimensional data often suffers from redundancy.  
Understanding how much of the essential information can be preserved in a low-dimensional latent space is critical for:
- **Data compression**
- **Efficient storage**
- **Noise reduction**
- **Feature engineering**

This project bridges **dimensionality reduction** with **representation learning**, revealing how a basic neural decoder can unlock the latent structures hidden in compressed data.

---

## 🧠 Future Work

- Experiment with non-linear dimensionality reduction techniques (e.g., t-SNE, UMAP).
- Implement more sophisticated decoders (e.g., Convolutional Autoencoders).
- Extend the approach to colored images and larger datasets.

---

## 📢 Acknowledgements

This project builds upon fundamental concepts of **unsupervised learning**, **representation learning**, and **neural network-based decoding** within the machine learning pipeline.

---

# 🔥 Academic Keywords

> PCA, Dimensionality Reduction, Neural Network Reconstruction, MLP Regressor, Latent Space, MNIST Dataset, Feature Compression, Representation Learning

---

