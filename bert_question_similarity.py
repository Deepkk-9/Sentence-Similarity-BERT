import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import openai

# Step 1: Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# Step 2: Function to get sentence embedding
def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
    return cls_embedding

# Step 3: Sample student questions
questions = [
    "I don't understand how photosynthesis works.",
    "Can someone explain the process of photosynthesis?",
    "What is the capital of France?",
    "Tell me about the capital city of France.",
    "How does the water cycle function?",
    "Explain the water cycle in detail.",
    "I like maths and science.",
    "How are maths and science related?"
]

# Step 4: Generate embeddings
embeddings = []
for question in questions:
    embedding = get_sentence_embedding(question)
    embeddings.append(embedding)
embeddings = np.vstack(embeddings)

# Step 5: Calculate cosine similarity matrix
similarity_matrix = cosine_similarity(embeddings)
print("Similarity Matrix:\n", similarity_matrix)

# Step 6: Cluster similar questions
k = 3  # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(embeddings)
labels = kmeans.labels_

# Step 7: Create DataFrame
df = pd.DataFrame({
    'Question': questions,
    'Cluster': labels
})
print(df)

# Step 8: (Optional) Visualize clusters
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)
plt.figure(figsize=(10, 7))
for i in range(k):
    cluster = reduced_embeddings[labels == i]
    plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {i}')
plt.title('Question Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()

# Step 9: Integrate with ChatGPT API for answers
openai.api_key = 'org-YYpS6uWuLu74XPA682Jp136x'  # Replace with your OpenAI API key

def get_chatgpt_answer(question):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=question,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Get answers for representative questions
representative_questions = df.groupby('Cluster')['Question'].first().tolist()
answers = [get_chatgpt_answer(q) for q in representative_questions]

# Assign answers to all questions based on cluster
df['Answer'] = df['Cluster'].apply(lambda x: answers[x])
print(df)


