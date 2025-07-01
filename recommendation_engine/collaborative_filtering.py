import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFiltering:
    def __init__(self, interactions_df=None):
        self.interactions_df = interactions_df
        self.user_item_matrix = None
        self.user_similarity = None

    def build_user_item_matrix(self):
        if self.interactions_df is None:
            print("Interaction data not provided.")
            return False

        self.user_item_matrix = self.interactions_df.pivot_table(
            index='user_id', columns='content_id', values='rating'
        ).fillna(0)
        return True

    def compute_user_similarity(self):
        if self.user_item_matrix is None:
            print("User-item matrix not built.")
            return False

        self.user_similarity = cosine_similarity(self.user_item_matrix)
        self.user_similarity = pd.DataFrame(
            self.user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        return True

    def fit(self, method='user-based'):
        if method == 'user-based':
            self.build_user_item_matrix()
            self.compute_user_similarity()

    def recommend_for_user(self, user_id, n_recommendations=5, exclude_watched=True):
        if self.user_similarity is None or self.user_item_matrix is None:
            print("Model not prepared. Run build_user_item_matrix() and compute_user_similarity() first.")
            return []

        if user_id not in self.user_similarity.index:
            print(f"User ID {user_id} not found.")
            return []

        sim_scores = self.user_similarity[user_id].drop(user_id)
        similar_users = sim_scores.sort_values(ascending=False)

        weighted_scores = np.zeros(self.user_item_matrix.shape[1])
        similarity_sums = np.zeros(self.user_item_matrix.shape[1])

        for sim_user, sim_score in similar_users.iteritems():
            sim_user_ratings = self.user_item_matrix.loc[sim_user].values
            weighted_scores += sim_user_ratings * sim_score
            similarity_sums += (sim_user_ratings > 0) * sim_score

        scores = np.divide(weighted_scores, similarity_sums, out=np.zeros_like(weighted_scores), where=similarity_sums != 0)
        recommendations = pd.Series(scores, index=self.user_item_matrix.columns)

        user_rated = self.user_item_matrix.loc[user_id]
        if exclude_watched:
            unseen_content = user_rated[user_rated == 0].index
        else:
            unseen_content = self.user_item_matrix.columns

        recommendations = recommendations[unseen_content].sort_values(ascending=False)
        return recommendations.head(n_recommendations).index.tolist()
