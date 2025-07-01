import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import mysql.connector
from mysql.connector import Error

class DataProcessor:
    def __init__(self, database_config=None):
        """
        Initialize the DataProcessor with optional database configuration.
        If database_config is None, it will work with CSV files instead.
        """
        self.database_config = database_config
        self.users_df = None
        self.content_df = None
        self.interactions_df = None
        
    def load_data_from_csv(self, users_path, content_path, interactions_path):
        """Load data from CSV files."""
        try:
            self.users_df = pd.read_csv(users_path)
            self.content_df = pd.read_csv(content_path)
            self.interactions_df = pd.read_csv(interactions_path)
            return True
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            return False
    
    def load_data_from_db(self):
        """Load data from MySQL database."""
        if not self.database_config:
            print("Database configuration not provided.")
            return False
        
        try:
            conn = mysql.connector.connect(**self.database_config)
            if conn.is_connected():
                self.users_df = pd.read_sql("SELECT * FROM users", conn)
                self.content_df = pd.read_sql("SELECT * FROM content", conn)
                self.interactions_df = pd.read_sql("SELECT * FROM interactions", conn)
                conn.close()
                return True
            else:
                print("Failed to connect to database.")
                return False
        except Error as e:
            print(f"Database error: {e}")
            return False
    
    def clean_data(self):
        """Clean and preprocess the data."""
        if self.users_df is None or self.content_df is None or self.interactions_df is None:
            print("Data not loaded. Please load data first.")
            return False
        
        # Remove duplicates
        self.users_df = self.users_df.drop_duplicates(subset=['user_id'])
        self.content_df = self.content_df.drop_duplicates(subset=['content_id'])
        self.interactions_df = self.interactions_df.drop_duplicates(subset=['interaction_id'])
        
        # Handle missing values
        self.users_df = self.users_df.fillna({'age': self.users_df['age'].median(), 
                                             'gender': 'Unknown',
                                             'location': 'Unknown'})
        
        self.content_df = self.content_df.fillna({'tags': '', 
                                                 'author': 'Unknown',
                                                 'popularity_score': self.content_df['popularity_score'].median()})
        
        self.interactions_df = self.interactions_df.fillna({'duration_seconds': 0, 
                                                           'rating': self.interactions_df['rating'].median()})
        
        # Convert timestamp to datetime
        if 'timestamp' in self.interactions_df.columns:
            self.interactions_df['timestamp'] = pd.to_datetime(self.interactions_df['timestamp'])
        
        return True
    
    def create_user_item_matrix(self):
        """Create a user-item interaction matrix for collaborative filtering."""
        if self.interactions_df is None:
            print("Interaction data not loaded.")
            return None
        
        # Create a user-item matrix with ratings
        user_item_matrix = self.interactions_df.pivot_table(
            index='user_id', 
            columns='content_id', 
            values='rating',
            aggfunc='mean',
            fill_value=0
        )
        
        return user_item_matrix
    
    def create_content_features(self):
        """Extract and normalize content features for content-based filtering."""
        if self.content_df is None:
            print("Content data not loaded.")
            return None
        
        # One-hot encode the content type and category
        content_type_dummies = pd.get_dummies(self.content_df['type'], prefix='type')
        content_category_dummies = pd.get_dummies(self.content_df['category'], prefix='category')
        
        # Process tags (create tag presence features)
        all_tags = set()
        for tags in self.content_df['tags'].str.split(',').dropna():
            all_tags.update([tag.strip() for tag in tags])
        
        tag_features = pd.DataFrame(0, index=self.content_df.index, columns=list(all_tags))
        
        for i, tags in enumerate(self.content_df['tags'].str.split(',').dropna()):
            for tag in tags:
                tag = tag.strip()
                if tag in all_tags:
                    tag_features.loc[i, tag] = 1
        
        # Normalize popularity score
        scaler = MinMaxScaler()
        popularity_normalized = pd.DataFrame(
            scaler.fit_transform(self.content_df[['popularity_score']]),
            columns=['normalized_popularity'],
            index=self.content_df.index
        )
        
        # Combine all features
        content_features = pd.concat([
            content_type_dummies, 
            content_category_dummies, 
            tag_features,
            popularity_normalized
        ], axis=1)
        
        # Add content_id as a column for reference
        content_features['content_id'] = self.content_df['content_id'].values
        
        return content_features
    
    def get_user_profile(self, user_id):
        """Get the profile of a specific user."""
        if self.users_df is None:
            print("User data not loaded.")
            return None
        
        return self.users_df[self.users_df['user_id'] == user_id].iloc[0] if len(self.users_df[self.users_df['user_id'] == user_id]) > 0 else None
    
    def get_content_details(self, content_ids):
        """Get details of specific content items."""
        if self.content_df is None:
            print("Content data not loaded.")
            return None
        
        return self.content_df[self.content_df['content_id'].isin(content_ids)]
    
    def get_user_interactions(self, user_id):
        """Get all interactions for a specific user."""
        if self.interactions_df is None:
            print("Interaction data not loaded.")
            return None
        
        return self.interactions_df[self.interactions_df['user_id'] == user_id]