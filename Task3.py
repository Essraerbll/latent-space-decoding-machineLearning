import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

# 1. Load the MNIST dataset from OpenML
# This dataset contains 70,000 handwritten digit images with 784 features (28x28 pixels).
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data  # Feature matrix: 70,000 samples and 784 features
y = mnist.target.astype(int)  # Labels for each sample, converted to integer format

# 2. Randomly sample 1,000 images while maintaining class distribution
# Stratified sampling ensures the class distribution remains proportional to the original dataset.
X_sampled, _, y_sampled, _ = train_test_split(X, y, train_size=1000, stratify=y, random_state=42)

# Standardize the features to have zero mean and unit variance for better PCA performance
scaler = StandardScaler()
X_sampled_scaled = scaler.fit_transform(X_sampled)

# 3. Use PCA for dimensionality reduction
# Reduce the dimensionality of the data from 784 features to 3 components (latent space).
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_sampled_scaled)

# 4. Create a Multi-Layer Perceptron (MLP) model for decoding
# The MLP will take the compressed data (3 components from PCA) and reconstruct the original 784 features.
mlp = MLPRegressor(hidden_layer_sizes=(10, 50), max_iter=500, random_state=42)

# 5. Train the MLP model with different iteration counts and store reconstructed images
iterations = [10, 25, 500]  # Training with 10, 25, and 500 iterations to observe performance
reconstructed_images = {}  # Dictionary to store reconstructed images for each iteration count

for iter_count in iterations:
    mlp.max_iter = iter_count  # Set the maximum number of iterations for training
    mlp.fit(X_pca, X_sampled_scaled)  # Train the MLP on the reduced data
    reconstructed = mlp.predict(X_pca)  # Reconstruct the original features from the PCA components
    reconstructed_images[iter_count] = reconstructed  # Store the reconstructed images

# 6. Visualize the original and reconstructed images for comparison
def plot_comparison(original_images, reconstructed_images, iterations):
    """
    Plot the original images and their reconstructed counterparts for different iteration counts.
    Args:
        original_images: List of tuples (image data, label) for original images.
        reconstructed_images: Dictionary of reconstructed images for each iteration count.
        iterations: List of iteration counts used during training.
    """
    # Create a figure with rows equal to the number of images and columns for original + iterations
    fig, axes = plt.subplots(len(original_images), len(iterations) + 1, figsize=(15, 15))

    # Add titles for the columns
    axes[0, 0].set_title("Original\nImages", fontsize=14, pad=30)  # First column: Original images
    for j, iter_count in enumerate(iterations):
        axes[0, j + 1].set_title(f"Reconstructed\nIter = {iter_count}", fontsize=12, pad=30)

    # Plot each original image and its reconstructions
    for i, (original, label) in enumerate(original_images):
        # Display the original image in the first column
        axes[i, 0].imshow(original.reshape(28, 28), cmap='gray')
        axes[i, 0].axis('off')  # Hide axis ticks
        axes[i, 0].set_ylabel(f"Label: {label}", fontsize=12)  # Add label for the original image

        # Display reconstructed images for each iteration count in subsequent columns
        for j, iter_count in enumerate(iterations):
            reconstructed = reconstructed_images[iter_count][i].reshape(28, 28)
            axes[i, j + 1].imshow(reconstructed, cmap='gray')
            axes[i, j + 1].axis('off')  # Hide axis ticks

    # Adjust layout to prevent overlap of titles and images
    plt.tight_layout()
    plt.show()

# Select 10 random images from the sampled dataset for visualization
original_images = [(X_sampled.iloc[i].values, y_sampled.iloc[i]) for i in range(10)]

# Plot the original images alongside their reconstructions
plot_comparison(original_images, reconstructed_images, iterations)
