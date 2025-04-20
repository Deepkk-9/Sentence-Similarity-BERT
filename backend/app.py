from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd 
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import hdbscan
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.io as pio 
import plotly.graph_objects as go
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import scipy.spatial
import os

# Import NLTK for linguistic analysis
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import RegexpParser

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Load Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Set matplotlib backend
plt.switch_backend('Agg')

app = Flask(__name__)
CORS(app)

def analyze_sentence_structure(question):
    """Analyze the linguistic structure of a question using NLTK"""
    tokens = word_tokenize(question)
    tagged = pos_tag(tokens)
    
    grammar = r"""
        NP: {<DT>?<JJ>*<NN.*>+}    # Noun phrases
        VP: {<VB.*><NP|PP|RB>*}    # Verb phrases
        PP: {<IN><NP>}             # Prepositional phrases
        QP: {<WP|WRB><VP>}         # Question phrases
    """
    parser = RegexpParser(grammar)
    result = parser.parse(tagged)
    
    features = {
        'complexity': len(list(result.subtrees())) - 1,
        'np_count': sum(1 for st in result.subtrees() if st.label() == 'NP'),
        'vp_count': sum(1 for st in result.subtrees() if st.label() == 'VP'),
        'qp_count': sum(1 for st in result.subtrees() if st.label() == 'QP'),
        'word_count': len(tokens)
    }
    
    features['linguistic_score'] = (features['complexity'] * 0.4 + 
                                   features['np_count'] * 0.2 + 
                                   features['vp_count'] * 0.2 + 
                                   features['qp_count'] * 0.2)
    return features

def generate_clusters(questions, clustering_method):
    """Function to generate clusters based on the chosen method"""
    # Generate embeddings (used for all methods except pure LDA)
    embeddings = model.encode(questions)
    
    if clustering_method == "lda":
        return lda_clustering(questions, embeddings)
    
    # For other clustering methods (DBSCAN, HDBSCAN, Agglomerative)
    cosine_sim_matrix = cosine_similarity(embeddings)
    distance_matrix = 1 - np.clip(cosine_sim_matrix, 0, 1)

    if clustering_method == "dbscan":
        cluster_model = DBSCAN(eps=0.45, min_samples=1, metric='precomputed')
    elif clustering_method == "hdbscan":
        cluster_model = hdbscan.HDBSCAN(min_cluster_size=2, metric='precomputed', cluster_selection_method='eom')
    elif clustering_method == "agglomerative":
        cluster_model = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average', distance_threshold=0.5)
    else:
        return jsonify({"error": "Invalid clustering method"}), 400

    clusters = cluster_model.fit_predict(distance_matrix.astype(np.float64))
    return prepare_response(questions, embeddings, clusters)

def lda_clustering(questions, embeddings):
    """Pure LDA clustering with dependency parsing representatives"""
    # LDA Topic Modeling
    vectorizer = TfidfVectorizer(stop_words='english', min_df=2)
    tfidf = vectorizer.fit_transform(questions)
    n_topics = min(5, max(1, len(questions)//3))
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_topics = lda.fit_transform(tfidf)
    clusters = np.argmax(lda_topics, axis=1)
    
    # Get topic keywords
    topic_keywords = {}
    for topic_id in range(n_topics):
        top_features = lda.components_[topic_id].argsort()[-10:][::-1]
        topic_keywords[topic_id] = [vectorizer.get_feature_names_out()[i] for i in top_features]
    
    # Prepare response (same format as other methods)
    response = prepare_response(questions, embeddings, clusters)
    
    # Add topic keywords to each cluster
    for label in response['clusters']:
        response['clusters'][label]['topic_keywords'] = topic_keywords.get(label, [])
    
    return response  # Return the response as a dictionary

def prepare_response(questions, embeddings, clusters):
    """Standardized response format for all clustering methods"""
    response = {
        'clusters': {},
        'intra_distances': {},
        'inter_distances': {}
    }
    
    excel_data = []
    unique_labels = set(clusters)
    
    for label in unique_labels:
        label = int(label)
        cluster_questions = [q for i, q in enumerate(questions) if clusters[i] == label]
        cluster_embeddings = embeddings[clusters == label]

        if cluster_questions:
            # Select representative using dependency parsing only
            representative = get_representative_by_parsing(cluster_questions)
            
            intra_distance = calculate_intra_cluster_distance(cluster_embeddings)
            centroid = np.mean(cluster_embeddings, axis=0)
            distances_from_centroid = np.linalg.norm(cluster_embeddings - centroid, axis=1)

            response['clusters'][label] = {
                'questions': cluster_questions,
                'representative_question': representative,
                'distances_from_centroid': distances_from_centroid.tolist()
            }
            response['intra_distances'][label] = float(intra_distance)

            for q in cluster_questions:
                excel_data.append([q, label, representative])

    # Save to Excel and generate visualization
    pd.DataFrame(excel_data, columns=['Question', 'Cluster Label', 'Representative Question']).to_excel('clusters.xlsx', index=False)
    response['inter_distances'] = calculate_inter_cluster_distance(embeddings, clusters)
    plot_clusters(embeddings, clusters, questions)
    
    return response

def get_representative_by_parsing(cluster_questions):
    """Select best question using dependency parsing"""
    if len(cluster_questions) == 1:
        return cluster_questions[0]
    
    best_question = None
    best_score = -1
    
    for q in cluster_questions:
        features = analyze_sentence_structure(q)
        score = features['linguistic_score']
        score -= 0.1 * abs(features['word_count'] - 12) ** 2  # Length penalty
        
        if score > best_score:
            best_score = score
            best_question = q
            
    return best_question

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

@app.route('/cluster/lda', methods=['POST'])
def cluster_lda():
    data = request.json
    questions = data.get('questions', [])
    return generate_clusters(questions, "lda")


def get_representative_question(cluster_questions, cluster_embeddings):
    """Find the question closest to the centroid"""
    centroid = np.mean(cluster_embeddings, axis=0)
    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
    closest_index = np.argmin(distances)
    return cluster_questions[closest_index]

def get_representative_question_using_dependency(cluster_questions, cluster_embeddings):
    """Select representative question using linguistic structure and embeddings"""
    if len(cluster_questions) == 1:
        return cluster_questions[0]
    
    # Calculate scores based on syntactic structure and centrality
    question_scores = []
    
    for i, question in enumerate(cluster_questions):
        # Get linguistic features
        features = analyze_sentence_structure(question)
        linguistic_score = features['linguistic_score']
        
        # Calculate embedding centrality
        centroid = np.mean(cluster_embeddings, axis=0)
        embedding_similarity = np.dot(cluster_embeddings[i], centroid) / (
            np.linalg.norm(cluster_embeddings[i]) * np.linalg.norm(centroid)
        )
        
        # Combine scores (linguistic structure and embedding similarity)
        # Higher linguistic_score means more well-structured question
        # Higher embedding_similarity means more central to the topic
        combined_score = (0.6 * linguistic_score) + (0.4 * embedding_similarity)
        question_scores.append(combined_score)
    
    # Select question with highest combined score
    best_question_index = np.argmax(question_scores)
    return cluster_questions[best_question_index]

def get_representative_question_using_lda(cluster_questions, cluster_embeddings):
    """Select the representative question using LDA and linguistic structure analysis"""
    if len(cluster_questions) == 1:
        return cluster_questions[0]
    
    # Step 1: Create a document-term matrix with TF-IDF weighting
    vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
    tfidf_matrix = vectorizer.fit_transform(cluster_questions)
    
    # Step 2: Train LDA model on the TF-IDF matrix
    lda_model_local = LatentDirichletAllocation(
        n_components=min(3, len(cluster_questions)), 
        random_state=42
    )
    lda_topics = lda_model_local.fit_transform(tfidf_matrix)
    
    # Step 3: Analyze importance of each question
    question_scores = []
    
    for i, question in enumerate(cluster_questions):
        # Get LDA topic distribution
        topic_dist = lda_topics[i]
        dominant_topic = np.argmax(topic_dist)
        topic_strength = topic_dist[dominant_topic]
        
        # Get linguistic features using NLTK
        features = analyze_sentence_structure(question)
        linguistic_score = features['linguistic_score'] 
        
        # Calculate embedding centrality
        centroid = np.mean(cluster_embeddings, axis=0)
        embedding_similarity = np.dot(cluster_embeddings[i], centroid) / (
            np.linalg.norm(cluster_embeddings[i]) * np.linalg.norm(centroid)
        )
        
        # Combine all scores
        combined_score = (0.4 * topic_strength) + (0.3 * linguistic_score) + (0.3 * embedding_similarity)
        question_scores.append(combined_score)
    
    # Select question with highest combined score
    best_question_index = np.argmax(question_scores)
    return cluster_questions[best_question_index]

def calculate_intra_cluster_distance(cluster_embeddings):
    """Calculate the average distance within a cluster"""
    if len(cluster_embeddings) > 1:
        pairwise_dists = pairwise_distances(cluster_embeddings)
        avg_intra_distance = np.mean(pairwise_dists[np.triu_indices(len(cluster_embeddings), k=1)])
        return avg_intra_distance
    return 0.0

def calculate_inter_cluster_distance(embeddings, clusters):
    """Calculate the distance between centroids of different clusters"""
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
    """Visualize clusters in 3D space"""
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    df = pd.DataFrame(reduced_embeddings, columns=['x', 'y', 'z'])
    df['Cluster'] = clusters
    df['Question'] = questions  # Add actual questions for hover text

    # Assign unique colors to clusters (and gray for noise)
    unique_labels = set(clusters)
    color_map = {label: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] 
                 for i, label in enumerate(unique_labels)}
    color_map[-1] = 'gray'  # Noise points in gray

    # Create figure
    fig = go.Figure()

    # Add scatter points for clusters
    for label in unique_labels:
        cluster_data = df[df['Cluster'] == label]
        fig.add_trace(go.Scatter3d(
            x=cluster_data['x'], y=cluster_data['y'], z=cluster_data['z'],
            mode='markers',
            marker=dict(
                size=8 if label != -1 else 5,
                opacity=0.85 if label != -1 else 0.5,
                color=color_map[label],
                line=dict(width=0.5, color='black')
            ),
            name=f'Cluster {label}' if label != -1 else 'Noise',
            hoverinfo='text',
            text=[f"Cluster: {label}<br>Question: {q}" for q in cluster_data['Question']]
        ))

        # Create Convex Hull for Each Cluster (Encircle the Points)
        if label != -1 and len(cluster_data) >= 4:  # Convex Hull needs at least 4 points
            try:
                hull = scipy.spatial.ConvexHull(cluster_data[['x', 'y', 'z']])
                hull_vertices = hull.vertices
                hull_points = cluster_data.iloc[hull_vertices]

                # Add mesh (Transparent Surface)
                fig.add_trace(go.Mesh3d(
                    x=hull_points['x'],
                    y=hull_points['y'],
                    z=hull_points['z'],
                    opacity=0.15,  # Make it slightly transparent
                    color=color_map[label],  # Use the cluster color
                    alphahull=0  # Convex hull shape
                ))
            except:
                # Skip hull generation if there's an error (like points being coplanar)
                pass

    # Adjust layout for better presentation
    fig.update_layout(
        title="3D Clustering with Enclosed Boundaries",
        scene=dict(
            xaxis_title="t-SNE Dim 1",
            yaxis_title="t-SNE Dim 2",
            zaxis_title="t-SNE Dim 3",
            bgcolor="white",  # White background
            xaxis=dict(backgroundcolor="white", gridcolor="lightgray"),
            yaxis=dict(backgroundcolor="white", gridcolor="lightgray"),
            zaxis=dict(backgroundcolor="white", gridcolor="lightgray")
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    # Set a dynamic camera angle
    fig.update_layout(scene_camera=dict(
        eye=dict(x=1.5, y=1.5, z=1.5)
    ))

    # Save as HTML file
    pio.write_html(fig, file='cluster_visualization.html', auto_open=True)
    return fig

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)