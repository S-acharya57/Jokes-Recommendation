import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns


data_path = "../data/"
ratings_df = pd.read_csv(data_path + "jester_ratings.csv")
jokes_df = pd.read_csv(data_path + "jester_items.csv")
# display(ratings_df.head(3))


# 1. Feature Extraction using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(jokes_df['jokeText'])

# LDA Topic Modeling
lda = LatentDirichletAllocation(n_components=2, random_state=0)
lda_topics = lda.fit_transform(tfidf_matrix)


# Combine TF-IDF and LDA topics into a single feature set
features = np.hstack([tfidf_matrix.toarray(), lda_topics])
# features.shape 

# Normalize features for cosine similarity calculation
scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features)

# Create a mapping from jokeId to index in the TF-IDF matrix
joke_id_to_index = {joke_id: index for index, joke_id in enumerate(jokes_df['jokeId'])}



# 2. Weighted Profile Building
def build_user_profile(user_id):

    
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    
    # Weight TF-IDF vectors by ratings
    weighted_tfidf_vectors = []
    for _, row in user_ratings.iterrows():
        joke_idx = joke_id_to_index[row['jokeId']]
        weighted_tfidf_vectors.append(features_normalized[joke_idx] * row['rating'])
    
    # Create user profile by averaging weighted TF-IDF vectors
    user_profile = np.mean(weighted_tfidf_vectors, axis=0)
    
    return user_profile


# 3. Similarity Calculation and Recommendations
def recommend_jokes(user_id, top_n=5):
    user_profile = build_user_profile(user_id)
    
    # Calculate cosine similarity between the user profile and all jokes
    similarities = cosine_similarity([user_profile], features_normalized)
    
    # Get the indices of the top_n most similar jokes
    similar_jokes_indices = similarities.flatten().argsort()[-top_n:]
    
    # Map indices back to jokeIds
    similar_jokes = jokes_df.iloc[similar_jokes_indices][['jokeId', 'jokeText']]
    similar_jokes['similarity'] = similarities.flatten()[similar_jokes_indices]
    
    return similar_jokes


def evaluate(user_id, top_n=5):
    
    recommended_jokes = recommend_jokes(user_id, top_n)
    
    return recommended_jokes