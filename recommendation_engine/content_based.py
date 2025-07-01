import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re

class ContentBasedFiltering:
    def __init__(self, content_df=None, interactions_df=None):
        """
        Initialize the content-based filtering recommender.
        
        Parameters:
        content_df (pandas.DataFrame): DataFrame with content features
        interactions_df (pandas.DataFrame): DataFrame with user-content interactions
        """
        self.content_df = content_df
        self.interactions_df = interactions_df
        self.content_similarity = None
        self.content_features = None
        self.tfidf_matrix = None
    
    def process_content_features(self):
        """Process and extract content features for similarity calculation."""
        if self.content_df is None:
            print("Content data not provided.")
            return False
        
        # Create a text representation of content features
        self.content_df['content_text'] = (
            self.content_df['title'] + ' ' + 
            self.content_df['type'] + ' ' + 
            self.content_df['category'] + ' ' + 
            self.content_df['tags'].str.replace(',', ' ') + ' ' + 
            self.content_df['author']
        )
        
        # Apply TF-IDF vectorization
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.content_df['content_text'])
        
        # Calculate content similarity matrix
        self.content_similarity = cosine_similarity(self.tfidf_matrix)
        
        return True
    
    def create_user_profiles(self):
        """Create content preference profiles for each user based on their interactions."""
        if self.interactions_df is None or self.tfidf_matrix is None:
            print("Interaction data or content features not available.")
            return {}
        
        user_profiles = {}
        
        # Group interactions by user
        user_groups = self.interactions_df.groupby('user_id')
        
        for user_id, interactions in user_groups:
            # Calculate a weighted average of content features based on user ratings
            user_content_indices = []
            user_content_weights = []
            
            for _, interaction in interactions.iterrows():
                content_id = interaction['content_id']
                # Find the index of this content in content_df
                content_idx = self.content_df.index[self.content_df['content_id'] == content_id].tolist()
                
                if content_idx:
                    content_idx = content_idx[0]
                    rating = interaction['rating']
                    user_content_indices.append(content_idx)
                    user_content_weights.append(rating)
            
            if user_content_indices:
                # Normalize weights
                user_content_weights = np.array(user_content_weights) / sum(user_content_weights)
                
                # Create user profile as weighted average of content features
                user_profile = np.zeros(self.tfidf_matrix.shape[1])
                for idx, weight in zip(user_content_indices, user_content_weights):
                    user_profile += weight * self.tfidf_matrix[idx].toarray().flatten()
                
                user_profiles[user_id] = user_profile
        
        return user_profiles
    
    def recommend_similar_content(self, content_id, n_recommendations=5):
        """
        Recommend content similar to a given content item.
        
        Parameters:
        content_id (int): ID of the content item
        n_recommendations (int): Number of recommendations to generate
        
        Returns:
        list: List of recommended content IDs
        """
        if self.content_similarity is None:
            print("Content similarity not calculated. Please run process_content_features() first.")
            return []
        
        # Find the index of the content item
        content_idx = self.content_df.index[self.content_df['content_id'] == content_id].tolist()
        
        if not content_idx:
            print(f"Content ID {content_id} not found.")
            return []
        
        content_idx = content_idx[0]
        
        # Get similarity scores for this content with all other content
        sim_scores = list(enumerate(self.content_similarity[content_idx]))
        
        # Sort content by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N+1 (excluding the input content itself)
        sim_scores = sim_scores[1:n_recommendations+1]
        
        # Get content indices
        content_indices = [i[0] for i in sim_scores]
        
        # Return content IDs
        return self.content_df.iloc[content_indices]['content_id'].tolist()
    
    def recommend_for_user(self, user_id, user_profiles=None, n_recommendations=5, exclude_watched=True):
        """
        Recommend content for a user based on their profile.
        
        Parameters:
        user_id (int): The ID of the user
        user_profiles (dict): Dictionary mapping user IDs to their content preference profiles
        n_recommendations (int): Number of recommendations to generate
        exclude_watched (bool): Whether to exclude content the user has already interacted with
        
        Returns:
        list: List of recommended content IDs
        """
        if user_profiles is None or user_id not in user_profiles:
            print(f"No profile found for user {user_id}.")
            return []
        
        # Get user profile
        user_profile = user_profiles[user_id].reshape(1, -1)
        
        # Calculate similarity between user profile and all content
        user_content_similarity = cosine_similarity(user_profile, self.tfidf_matrix).flatten()
        
        # Create a list of (content_idx, similarity) tuples
        content_sim_scores = list(enumerate(user_content_similarity))
        
        # Sort by similarity
        content_sim_scores = sorted(content_sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get content the user has already interacted with
        if exclude_watched and self.interactions_df is not None:
            user_interactions = self.interactions_df[self.interactions_df['user_id'] == user_id]
            user_content = set(user_interactions['content_id'])
        else:
            user_content = set()
        
        # Filter and get top recommendations
        recommendations = []
        for idx, _ in content_sim_scores:
            content_id = self.content_df.iloc[idx]['content_id']
            if content_id not in user_content:
                recommendations.append(content_id)
                if len(recommendations) >= n_recommendations:
                    break
        
        return recommendations
    
    def recommend_by_category(self, user_id, category, n_recommendations=5, exclude_watched=True):
        """
        Recommend content from a specific category based on user preferences.
        
        Parameters:
        user_id (int): The ID of the user
        category (str): The content category to recommend from
        n_recommendations (int): Number of recommendations to generate
        exclude_watched (bool): Whether to exclude content the user has already interacted with
        
        Returns:
        list: List of recommended content IDs
        """
        # Get content the user has already interacted with
        if exclude_watched and self.interactions_df is not None:
            user_interactions = self.interactions_df[self.interactions_df['user_id'] == user_id]
            user_content = set(user_interactions['content_id'])
        else:
            user_content = set()
        
        # Filter content by category
        category_content = self.content_df[self.content_df['category'] == category]
        
        if len(category_content) == 0:
            print(f"No content found in category '{category}'.")
            return []
        
        # If we have user interactions, recommend based on user preferences
        if self.interactions_df is not None and len(user_interactions) > 0:
            # Create user profiles
            user_profiles = self.create_user_profiles()
            
            if user_id in user_profiles:
                # Get user profile
                user_profile = user_profiles[user_id].reshape(1, -1)
                
                # Calculate similarity between user profile and category content
                category_indices = category_content.index
                category_features = self.tfidf_matrix[category_indices]
                
                user_content_similarity = cosine_similarity(user_profile, category_features).flatten()
                
                # Create a list of (content_idx, similarity) tuples
                content_sim_scores = [(idx, user_content_similarity[i]) 
                                    for i, idx in enumerate(category_indices)]
                
                # Sort by similarity
                content_sim_scores = sorted(content_sim_scores, key=lambda x: x[1], reverse=True)
                
                # Filter and get top recommendations
                recommendations = []
                for idx, _ in content_sim_scores:
                    content_id = self.content_df.iloc[idx]['content_id']
                    if content_id not in user_content:
                        recommendations.append(content_id)
                        if len(recommendations) >= n_recommendations:
                            break
                
                return recommendations
        
        # If no user interactions or profile, sort by popularity
        popular_content = category_content.sort_values('popularity_score', ascending=False)
        
        # Filter out content the user has already interacted with
        recommendations = []
        for _, content_row in popular_content.iterrows():
            content_id = content_row['content_id']
            if content_id not in user_content:
                recommendations.append(content_id)
                if len(recommendations) >= n_recommendations:
                    break
        
        return recommendations