import numpy as np
# Implement PCA
def pca(data, n_components):
    # Compute the covariance matrix
    covariance_matrix = np.cov(data.T)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top n_components
    selected_vectors = eigenvectors[:, :n_components]

    # Project the data onto the selected eigenvectors
    reduced_data = np.dot(data, selected_vectors)

    # Correct calculation of explained variance ratio
    total_variance = np.sum(eigenvalues)  # Sum of ALL eigenvalues
    explained_variance_ratio = eigenvalues[:n_components] / total_variance

    return reduced_data, explained_variance_ratio