from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd 
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import hdbscan
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

# Set matplotlib backend
plt.switch_backend('Agg')

app = Flask(__name__)
CORS(app)

# Load Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_clusters(questions, clustering_method):
    """Function to generate clusters based on the chosen method"""
    # Generate embeddings
    embeddings = model.encode(questions)

    # Compute the cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(embeddings)

    # Convert cosine similarity to distance
    distance_matrix = 1 - np.clip(cosine_sim_matrix, 0, 1)

    if clustering_method == "dbscan":
        cluster_model = DBSCAN(eps=0.65, min_samples=1, metric='precomputed')
    elif clustering_method == "hdbscan":
        cluster_model = hdbscan.HDBSCAN(min_cluster_size=2, metric='precomputed', cluster_selection_method='eom')
    elif clustering_method == "agglomerative":
        cluster_model = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average', distance_threshold=0.5)
    else:
        return jsonify({"error": "Invalid clustering method"}), 400

    clusters = cluster_model.fit_predict(distance_matrix.astype(np.float64))

    # Prepare response
    response = {
        'clusters': {},
        'intra_distances': {},
        'inter_distances': {}
    }

    # Data for Excel file
    excel_data = []

    unique_labels = set(clusters)
    for label in unique_labels:
        label = int(label)  # Convert to integer
        cluster_questions = [questions[i] for i in range(len(questions)) if clusters[i] == label]
        cluster_embeddings = embeddings[clusters == label]

        if cluster_questions:
            representative_question = get_representative_question(cluster_questions, cluster_embeddings)
            intra_distance = calculate_intra_cluster_distance(cluster_embeddings)
            response['intra_distances'][label] = float(intra_distance)

            centroid = np.mean(cluster_embeddings, axis=0)
            distances_from_centroid = np.linalg.norm(cluster_embeddings - centroid, axis=1)

            response['clusters'][label] = {
                'questions': cluster_questions,
                'representative_question': representative_question,
                'distances_from_centroid': distances_from_centroid.tolist()
            }

            # Add data to Excel file
            for q in cluster_questions:
                excel_data.append([q, label, representative_question])

    # Convert data to DataFrame and save to Excel
    df = pd.DataFrame(excel_data, columns=['Question', 'Cluster Label', 'Representative Question'])
    df.to_excel('clusters.xlsx', index=False)

    response['inter_distances'] = calculate_inter_cluster_distance(embeddings, clusters)
    plot_clusters(embeddings, clusters, questions)
    
    return jsonify(response)

@app.route('/cluster/dbscan', methods=['POST'])
def cluster_dbscan():
    data = request.json
    questions = data.get('questions', [])
    return generate_clusters(questions, "dbscan")

@app.route('/cluster/hdbscan', methods=['POST'])
def cluster_hdbscan():
    data = request.json
    questions = data.get('questions', [])
    return generate_clusters(questions, "hdbscan")

@app.route('/cluster/agglomerative', methods=['POST'])
def cluster_agglomerative():
    data = request.json
    questions = data.get('questions', [])
    return generate_clusters(questions, "agglomerative")

def get_representative_question(cluster_questions, cluster_embeddings):
    centroid = np.mean(cluster_embeddings, axis=0)
    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
    closest_index = np.argmin(distances)
    return cluster_questions[closest_index]

def calculate_intra_cluster_distance(cluster_embeddings):
    if len(cluster_embeddings) > 1:
        pairwise_dists = pairwise_distances(cluster_embeddings)
        avg_intra_distance = np.mean(pairwise_dists[np.triu_indices(len(cluster_embeddings), k=1)])
        return avg_intra_distance
    return 0.0

def calculate_inter_cluster_distance(embeddings, clusters):
    unique_labels = set(clusters)
    centroids = {label: np.mean(embeddings[clusters == label], axis=0) for label in unique_labels if label != -1}
    
    inter_distances = {}
    centroid_keys = list(centroids.keys())
    for i in range(len(centroid_keys)):
        for j in range(i + 1, len(centroid_keys)):
            dist = np.linalg.norm(centroids[centroid_keys[i]] - centroids[centroid_keys[j]])
            inter_distances[f"{centroid_keys[i]}-{centroid_keys[j]}"] = float(dist)

    return inter_distances

def plot_clusters(embeddings, clusters, questions):
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 6))
    unique_labels = set(clusters)
    
    for label in unique_labels:
        indices = np.where(clusters == label)[0]  # Correct indexing
        if label == -1:
            plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], color='red', label='Noise', s=100)
        else:
            plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], label=f'Cluster {label}', s=100)

    for i, question in enumerate(questions):
        plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], str(i), fontsize=9)

    plt.title('Question Clusters')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.grid(True)

    # Save the plot as an image file
    plt.savefig('clusters_plot.png')
    plt.close()


if __name__ == '__main__':
    app.run(debug=True)
