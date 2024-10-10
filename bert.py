import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# Function to read questions from a text file
def read_questions_from_file(filename):
    with open(filename, 'r') as file:
        questions = file.readlines()
    return [question.strip() for question in questions]

# Function to write output to a text file
def write_clusters_to_file(questions_df, cluster_groups, output_filename="clustered_questions.txt"):
    with open(output_filename, 'w') as file:
        file.write("Clustered Questions:\n")
        file.write(questions_df.to_string(index=False))
        file.write("\n\nCluster Representations:\n")
        for cluster_id, representative_question in cluster_groups.items():
            file.write(f"Cluster {cluster_id}: {representative_question}\n")

# Sample input text file (you can replace 'input_questions.txt' with your own file)
questions = read_questions_from_file('questions.txt')

# Convert the list of questions to a DataFrame
questions_df = pd.DataFrame(questions, columns=["Question"])

# Initialize the Sentence-BERT model to get embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Use Sentence-BERT for better sentence embeddings
question_embeddings = model.encode(questions_df['Question'].to_list())

# DBSCAN: No need to specify the number of clusters, clusters formed dynamically
dbscan = DBSCAN(eps=0.5, min_samples=2, metric="cosine").fit(question_embeddings)

# Assign the cluster labels to each question
questions_df['Cluster'] = dbscan.labels_

# Group by cluster and select a representative question for each cluster
cluster_groups = questions_df.groupby('Cluster')['Question'].apply(lambda x: x.iloc[0]).to_dict()

# Write the output to a text file (you can change the output file name if needed)
write_clusters_to_file(questions_df, cluster_groups, output_filename="clustered_questions.txt")

# Display the questions with their respective clusters
print(questions_df)

# Display representative question for each cluster
print("\nCluster Representations:")
for cluster_id, representative_question in cluster_groups.items():
    print(f"Cluster {cluster_id}: {representative_question}")
