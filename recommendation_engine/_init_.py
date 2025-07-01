from .collaborative_filtering import get_collaborative_recommendations
from .content_based import get_content_based_recommendations

def get_recommendations(user_id, n=5):
    """
    Get hybrid recommendations for a user
    
    Args:
        user_id (int): The user ID to get recommendations for
        n (int): Number of recommendations to return
    
    Returns:
        list: List of dictionaries containing recommended product information
    """
    # Get recommendations from both methods
    collab_recs = get_collaborative_recommendations(user_id, n)
    content_recs = get_content_based_recommendations(user_id, n)
    
    # Merge recommendations, prioritizing collaborative filtering results
    # but ensuring diversity with content-based results
    seen_product_ids = set()
    hybrid_recs = []
    
    # Add collaborative filtering recommendations first
    for rec in collab_recs:
        hybrid_recs.append(rec)
        seen_product_ids.add(rec['product_id'])
    
    # Add unique content-based recommendations
    for rec in content_recs:
        if rec['product_id'] not in seen_product_ids and len(hybrid_recs) < n:
            hybrid_recs.append(rec)
            seen_product_ids.add(rec['product_id'])
    
    # Sort by score
    hybrid_recs = sorted(hybrid_recs, key=lambda x: x['score'], reverse=True)
    
    # Return top n recommendations
    return hybrid_recs[:n]
