import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("tracks.csv", low_memory=False)

# Select and clean relevant features
music = df[['song', 'Genre', 'Popularity', 'emotion', 'Instrumentalness']].copy()
music.dropna(subset=['Genre', 'Popularity', 'emotion', 'Instrumentalness'], inplace=True)

# Convert genre string to list
music['genre_list'] = music['Genre'].apply(lambda x: [g.strip() for g in x.split(',')] if isinstance(x, str) else [])

# One-hot encode genre
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(music['genre_list'])
genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)

# One-hot encode emotion from dataset
emotion_encoded = pd.get_dummies(music['emotion'])

# Scale numerical features
scaler = MinMaxScaler()
music[['Popularity_scaled', 'Instrumentalness_scaled']] = scaler.fit_transform(
    music[['Popularity', 'Instrumentalness']]
)

# Combine all features for clustering and similarity
final_features = pd.concat([
    genre_df.reset_index(drop=True),
    emotion_encoded.reset_index(drop=True),
    music[['Popularity_scaled', 'Instrumentalness_scaled']].reset_index(drop=True)
], axis=1)

# Clustering with KMeans
optimal_k = 8
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
music['cluster'] = kmeans.fit_predict(final_features)

# Plot clusters
plt.figure(figsize=(10, 5))
sns.countplot(x=music['cluster'], palette='Set2')
plt.title('Number of Songs per Cluster (with Dataset Emotion)')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Multi-song recommendation function
def recommend_from_multiple_songs(song_list, n=10):
    song_list = [s.lower().strip() for s in song_list]
    selected = music[music['song'].str.lower().isin(song_list)]
    
    if selected.empty:
        return "None of the provided songs were found in the dataset."

    selected_vectors = final_features.loc[selected.index]
    user_profile = selected_vectors.mean(axis=0).values.reshape(1, -1)

    distances = euclidean_distances(final_features, user_profile).flatten()
    music['distance_to_user_profile'] = distances

    recommendations = music[~music['song'].str.lower().isin(song_list)]
    
    return recommendations.sort_values(by='distance_to_user_profile').head(n)[
        ['song', 'Genre', 'emotion', 'Popularity', 'Instrumentalness']
    ]

user_input = input("Enter your 3–5 favorite songs:\n")
user_songs = [song.strip() for song in user_input.split(',')]

print("\n🎵 Recommended Songs Based on Your Taste:\n")
recommendations = recommend_from_multiple_songs(user_songs, n=10)
print(recommendations)
