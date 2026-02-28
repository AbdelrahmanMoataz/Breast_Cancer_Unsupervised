import numpy as np
import random
# Step 1: Initialize cluster centroids
def choose_center(k, init, data, random_state=None):
    n_samples, n_features = data.shape
    centers = np.zeros((k, n_features))
    # Set random state if you want reproducable results
    if random_state is not None:
        np.random.seed(random_state) #seed numpy
        random.seed(random_state) #seed python random
        
    # Randomly choose data points as starting points for clusters with no replacement
    if init == 'random':
        centers = data[np.random.choice(data.shape[0], k, replace=False), :]
            
    # A slower algorithm that tries to make the starting clusters far away from each other to help K means converge faster
    if init == 'k-means++':
        # 1. Choose the first center randomly from the data points.
        centers[0] = data[np.random.randint(n_samples)]
    
        for i in range(1, k):
            # 2. For each data point x, compute D(x), the distance between x and the nearest center that has already been chosen.
            distances = np.zeros(n_samples)
            for j in range(n_samples):
                min_dist = float('inf')
                for l in range(i):  # Compare to already chosen centers
                    dist = np.linalg.norm(data[j] - centers[l])
                    min_dist = min(min_dist, dist)
                distances[j] = min_dist
    
            # 3. Choose one new data point as a new center, using a weighted probability distribution where a point x is chosen with probability proportional to D(x)^2.
            probabilities = distances**2 / np.sum(distances**2)
            cumulative_probabilities = np.cumsum(probabilities)
            random_value = random.random()
            for j in range(n_samples):
                if random_value < cumulative_probabilities[j]:
                    centers[i] = data[j]
                    break

    return centers
# Step 2: Assign data points to nearest cluster center
def assign_clusters(data, clusters):
    for sample in data:
        distance_to_clusters = []
        # length clusters is passed instead of K because it could change during Step 3
        for i in range(len(clusters)):
            # Calculate the squared Euclidean distance (L2 Norm)
            distance = np.sum((sample - clusters[i]['center']) ** 2)
            distance_to_clusters.append(distance)
        # Find index of cluster that resulted in smallest distance and assign the data point to it
        assigned_cluster = np.argmin(distance_to_clusters)
        clusters[assigned_cluster]['samples'].append(sample)
    return clusters
# Step 3: Update cluster centers positions
def update_clusters(clusters):
    # length clusters is passed instead of K because it could change during Step 3
    for i in range(len(clusters)):
        samples = np.array(clusters[i]['samples'])
        # Need to check if number of samples assigned to a cluster is greater than zero before calculating the mean
        if samples.shape[0] > 0:
            # axis=0 means that means of each feature are calculated separately
            new_center = samples.mean(axis=0)
            clusters[i]['center'] = new_center
            # Reset points of cluster for the next iteration
            clusters[i]['samples'] = []
        else:
            print(f'cluster {i+1} was deleted!')
            del clusters[i]
    return clusters
# Calculate the within cluster sum of squares (WCSS) or inertia
def calculate_inertia(clusters):
    squared_distance_per_cluster = {}
    squared_distances = 0
    for i in range(len(clusters)):
        center = clusters[i]['center']
        samples = np.array(clusters[i]['samples'])
        cluster_distance = np.sum((samples - center) ** 2)
        squared_distance_per_cluster[i] = cluster_distance
        squared_distances += cluster_distance
    return squared_distances, squared_distance_per_cluster

# Join all previous functions together
# init can be 'k-means++' or 'random' for initialization using random data points
# an init of 'random' requires a number of repititions to produce the best set of clusters
# K means++ doesn't require repititions since it should produce initial centers that are away from each other
from copy import deepcopy
def kmeans(data, k, init='k-means++', reps=50, max_iter=100, tolerance=1e-10, random_state=None):
    n_samples, n_features = data.shape
    if init == 'k-means++':
        reps = 3
    clusters_list = []
    # Repeat K means with multiple inital starts to get a list of different clusters
    for _ in range(reps):
        # Step 1: Initialize clusters
        clusters = {}
        centers = choose_center(k, init, data, random_state)
        for i in range(k):
            cluster = {
                'center' : centers[i],
                'samples' : []
            }
            clusters[i] = cluster
        # Now enter loop to find best center positions
        old_cost = float('inf')
        old_clusters = deepcopy(clusters) # Create deepcopy so changes in original don't reflect in new variable
        for c in range(max_iter):
            # Step 2: Assign data points to nearest cluster center
            clusters = assign_clusters(data, clusters)
            
            # Step 3: Update cluster centers positions
            clusters = update_clusters(clusters)
            
            # Check for convergence: 
            old_centers = np.array([old_clusters[cluster]['center'] for cluster in old_clusters])
            new_centers = np.array([clusters[cluster]['center'] for cluster in clusters])
            #print([old_clusters[cluster]['center'] for cluster in old_clusters])
            #print([clusters[cluster]['center'] for cluster in clusters])
            if np.all(np.abs(old_centers - new_centers) < tolerance):
                clusters_list.append(clusters)
                #print(c)
                break
            if c == max_iter - 1:
                clusters_list.append(clusters)
            old_clusters = deepcopy(clusters) # Create deepcopy so changes in original don't reflect in new variable
            
    # Find the best cluster by finding least cost
    best_cost = float('inf')
    best_clusters = {}
    for clusters in clusters_list:
        # After final iteration, points are not assigned to updated cluster positions yet and therefore must assign before calculating cost
        clusters = assign_clusters(data, clusters)
        cost, _ = calculate_inertia(clusters)
        #print(cost)
        if cost < best_cost:
            best_cost = cost
            best_clusters = clusters
    return best_cost, best_clusters