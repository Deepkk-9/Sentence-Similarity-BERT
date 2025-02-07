from flask import Flask, request, jsonify
import numpy as np
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Set the matplotlib backend to 'Agg' to avoid GUI issues
plt.switch_backend('Agg')

app = Flask(__name__)

# Load the Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/cluster', methods=['POST'])
def cluster_questions():
    data = request.json
    questions = data.get('questions', [])

    # Generate embeddings
    embeddings = model.encode(questions)

    # Compute the cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(embeddings)

    # Convert cosine similarity to distance (1 - similarity)
    distance_matrix = 1 - cosine_sim_matrix

    # Apply DBSCAN
    dbscan = DBSCAN(eps=0.9, min_samples=1)  # Adjust eps according to your data
    clusters = dbscan.fit_predict(distance_matrix)

    # Prepare response
    response = {
        'clusters': {},
        'intra_distances': {},
        'inter_distances': {}
    }

    # Handle clusters and noise
    unique_labels = set(clusters)
    for label in unique_labels:
        # Convert label to a regular Python integer
        label = int(label)  
        cluster_questions = [questions[i] for i in range(len(questions)) if clusters[i] == label]
        cluster_embeddings = embeddings[clusters == label]

        if cluster_questions:  # Ensure there are questions in the cluster
            # Find representative question for the current cluster
            representative_question = get_representative_question(cluster_questions, cluster_embeddings)

            # Calculate intra-cluster distance
            intra_distance = calculate_intra_cluster_distance(cluster_embeddings)
            response['intra_distances'][label] = float(intra_distance)  # Convert to Python float

            # Calculate centroid
            centroid = np.mean(cluster_embeddings, axis=0)

            # Calculate distances from centroid for each question
            distances_from_centroid = np.linalg.norm(cluster_embeddings - centroid, axis=1)

            # Add to response
            response['clusters'][label] = {
                'questions': cluster_questions,
                'representative_question': representative_question,
                'distances_from_centroid': distances_from_centroid.tolist()  # Convert to list for JSON serialization
            }

    # Calculate inter-cluster distances
    response['inter_distances'] = calculate_inter_cluster_distance(embeddings, clusters)

    # Plotting the clusters in 2D
    plot_clusters(embeddings, clusters, questions)

    return jsonify(response)


def get_representative_question(cluster_questions, cluster_embeddings):
    # Calculate the centroid of the embeddings
    centroid = np.mean(cluster_embeddings, axis=0)

    # Calculate distances from centroid to each embedding in the cluster
    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)

    # Get the index of the closest question to the centroid
    closest_index = np.argmin(distances)

    # Return the closest question
    return cluster_questions[closest_index]

def calculate_intra_cluster_distance(cluster_embeddings):
    """Calculate average intra-cluster distance."""
    if len(cluster_embeddings) > 1:
        pairwise_dists = pairwise_distances(cluster_embeddings)
        avg_intra_distance = np.mean(pairwise_dists[np.triu_indices(len(cluster_embeddings), k=1)])
        return avg_intra_distance
    return 0.0  # Only one point in the cluster

def calculate_inter_cluster_distance(embeddings, clusters):
    """Calculate inter-cluster distances between centroids."""
    unique_labels = set(clusters)
    centroids = {}
    
    for label in unique_labels:
        if label != -1:  # Exclude noise
            centroids[label] = np.mean(embeddings[clusters == label], axis=0)

    # Calculate distances between centroids
    inter_distances = {}
    centroid_keys = list(centroids.keys())
    for i in range(len(centroid_keys)):
        for j in range(i + 1, len(centroid_keys)):
            dist = np.linalg.norm(centroids[centroid_keys[i]] - centroids[centroid_keys[j]])
            # Convert tuple key to string and ensure float is standard Python float
            inter_distances[f"{centroid_keys[i]}-{centroid_keys[j]}"] = float(dist)

    return inter_distances

def plot_clusters(embeddings, clusters, questions):
    """Plot the clusters in 2D and annotate points with question labels."""
    # Reduce dimensions for plotting
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 6))

    # Plot each cluster
    unique_labels = set(clusters)
    for label in unique_labels:
        if label == -1:  # Plot noise
            noise_points = reduced_embeddings[clusters == -1]
            plt.scatter(noise_points[:, 0], noise_points[:, 1], color='red', label='Noise', s=100)
        else:  # Plot regular clusters
            cluster_points = reduced_embeddings[clusters == label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}', s=100)

    # Annotate points with their respective questions
    for i, question in enumerate(questions):
        plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], question, fontsize=9)

    plt.title('Question Clusters')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.grid(True)
    plt.savefig('clusters_plot.png')  # Save the plot to a file
    plt.close()  # Close the plot to free up memory

if __name__ == '__main__':
    app.run(debug=True)
