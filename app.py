import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from PIL import Image
import random
from recommendation_engine.data_processing import DataProcessor
from recommendation_engine.collaborative_filtering import CollaborativeFiltering
from recommendation_engine.content_based import ContentBasedFiltering

# Set page configuration
st.set_page_config(
    page_title="Content Recommendation System",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .content-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .recommended-tag {
        background-color: #4CAF50;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
    .article-type {
        background-color: #2196F3;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        font-size: 0.8rem;
    }
    .video-type {
        background-color: #F44336;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        font-size: 0.8rem;
    }
    .ad-type {
        background-color: #FF9800;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        font-size: 0.8rem;
    }
    .tag {
        background-color: #E0E0E0;
        color: #333;
        padding: 0.1rem 0.4rem;
        border-radius: 3px;
        font-size: 0.7rem;
        margin-right: 0.3rem;
    }
    .user-info {
        background-color: #f1f8e9;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 5px solid #8BC34A;
    }
    .stat-card {
        background-color: white;
        border-radius: 5px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #757575;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'selected_user' not in st.session_state:
    st.session_state.selected_user = None
if 'collaborative_model' not in st.session_state:
    st.session_state.collaborative_model = None
if 'content_model' not in st.session_state:
    st.session_state.content_model = None

# Function to load and process data
@st.cache_data
def load_data():
    # Define data paths
    data_dir = "data"
    users_path = os.path.join(data_dir, "users.csv")
    content_path = os.path.join(data_dir, "content.csv")
    interactions_path = os.path.join(data_dir, "interactions.csv")
    
    # Initialize data processor
    data_processor = DataProcessor()
    
    # Load data from CSV files
    success = data_processor.load_data_from_csv(users_path, content_path, interactions_path)
    
    if success:
        # Clean data
        data_processor.clean_data()
        
        # Initialize and train collaborative filtering model
        interactions_df = data_processor.interactions_df  # Pass this directly
        collaborative_model = CollaborativeFiltering(interactions_df)
        collaborative_model.fit(method='user-based')
        
        # Initialize and train content-based model
        content_model = ContentBasedFiltering(data_processor.content_df, data_processor.interactions_df)
        content_model.process_content_features()
        
        return data_processor, collaborative_model, content_model, True
    else:
        return None, None, None, False

# Function to display content item
def display_content_item(content_item, is_recommended=False):
    col1, col2 = st.columns([1, 3])
    
    # Display placeholder image based on content type
    with col1:
        content_type = content_item['type']
        if content_type == 'article':
            st.image("https://via.placeholder.com/150x100?text=Article", use_column_width=True)
        elif content_type == 'video':
            st.image("https://via.placeholder.com/150x100?text=Video", use_column_width=True)
        elif content_type == 'ad':
            st.image("https://via.placeholder.com/150x100?text=Advertisement", use_column_width=True)
    
    # Display content details
    with col2:
        title_html = f"{content_item['title']}"
        
        # Add recommended tag if applicable
        if is_recommended:
            title_html += ' <span class="recommended-tag">Recommended</span>'
        
        # Add content type tag
        if content_type == 'article':
            type_html = '<span class="article-type">Article</span>'
        elif content_type == 'video':
            type_html = '<span class="video-type">Video</span>'
        elif content_type == 'ad':
            type_html = '<span class="ad-type">Ad</span>'
        else:
            type_html = ''
        
        st.markdown(f"### {title_html} {type_html}", unsafe_allow_html=True)
        
        # Display tags
        tags_html = ""
        if not pd.isna(content_item['tags']):
            tags = content_item['tags'].split(',')
            for tag in tags:
                tags_html += f'<span class="tag">{tag.strip()}</span>'
        
        st.markdown(f"<div>{tags_html}</div>", unsafe_allow_html=True)
        
        # Display content info
        st.markdown(f"""
        **Category:** {content_item['category']}  
        **Author:** {content_item['author']}  
        **Published:** {content_item['publish_date']}  
        **Popularity Score:** {content_item['popularity_score']}
        """)
        
        # Add interaction buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.button("View", key=f"view_{content_item['content_id']}")
        with col2:
            st.button("Like", key=f"like_{content_item['content_id']}")

# Function to display user profile
def display_user_profile(user_profile, interactions_df, content_df):
    st.markdown('<div class="user-info">', unsafe_allow_html=True)
    
    # User basic info
    st.markdown(f"## User: {user_profile['username']}")
    st.markdown(f"""
    📍 **Location:** {user_profile['location']}  
    👤 **Age:** {user_profile['age']}  
    ⚥ **Gender:** {user_profile['gender']}  
    📆 **Joined:** {user_profile['signup_date']}
    """)
    
    # User interaction statistics
    user_interactions = interactions_df[interactions_df['user_id'] == user_profile['user_id']]
    viewed_content = content_df[content_df['content_id'].isin(user_interactions['content_id'])]
    
    # Count content types the user has interacted with
    content_type_counts = viewed_content['type'].value_counts()
    article_count = content_type_counts.get('article', 0)
    video_count = content_type_counts.get('video', 0)
    ad_count = content_type_counts.get('ad', 0)
    
    # Count content categories the user has interacted with
    top_categories = viewed_content['category'].value_counts().head(3)
    
    # Display interaction stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value">{len(user_interactions)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Total Interactions</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value">{article_count}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Articles Viewed</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value">{video_count}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Videos Watched</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value">{ad_count}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Ads Clicked</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display top categories
    if len(top_categories) > 0:
        st.markdown("### Top Categories")
        for category, count in top_categories.items():
            st.markdown(f"- **{category}**: {count} interactions")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main app layout
st.markdown('<h1 class="main-header">📱 Smart Content Recommendation System</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Home", "Recommendations", "Browse Content", "Analytics"])
    
    st.header("Settings")
    data_source = st.selectbox("Data Source", ["CSV Files", "MySQL Database"])
    
    # Load data button
    if st.button("Load Data"):
        with st.spinner("Loading data..."):
            data_processor, collaborative_model, content_model, success = load_data()
            if success:
                st.session_state.data_processor = data_processor
                st.session_state.collaborative_model = collaborative_model
                st.session_state.content_model = content_model
                st.session_state.data_loaded = True
                st.success("Data loaded successfully!")
            else:
                st.error("Failed to load data. Please check your data files.")
    
    # User selection (only shown after data is loaded)
    if st.session_state.data_loaded:
        st.header("User Selection")
        users_df = st.session_state.data_processor.users_df
        user_options = [(row['user_id'], row['username']) for _, row in users_df.iterrows()]
        selected_user_id = st.selectbox(
            "Select User", 
            options=[user[0] for user in user_options],
            format_func=lambda x: next((user[1] for user in user_options if user[0] == x), str(x))
        )
        
        if selected_user_id != st.session_state.selected_user:
            st.session_state.selected_user = selected_user_id

# Content based on selected page
if page == "Home":
    # Home page content
    st.markdown('<h2 class="section-header">Welcome to Smart Content Recommendation System</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        This system provides personalized recommendations for articles, videos, and ads based on user preferences and behavior.
        
        **Key Features:**
        - Personalized content recommendations
        - Multiple recommendation algorithms
        - User profile analysis
        - Content browsing by category
        - Analytics dashboard
        """)
    
    with col2:
        st.image("https://via.placeholder.com/600x400?text=Smart+Recommendation+System", use_column_width=True)
    
    st.markdown("### How to Get Started")
    st.markdown("""
    1. Load the data using the button in the sidebar
    2. Select a user from the dropdown menu
    3. Navigate to the 'Recommendations' page to see personalized content
    4. Explore other features through the navigation menu
    """)
    
    # Show a sample of content if data is loaded
    if st.session_state.data_loaded:
        st.markdown('<h2 class="section-header">Featured Content</h2>', unsafe_allow_html=True)
        data_processor = st.session_state.data_processor
        
        # Get a sample of popular content
        popular_content = data_processor.content_df.sort_values('popularity_score', ascending=False).head(3)
        
        for _, content_item in popular_content.iterrows():
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            display_content_item(content_item)
            st.markdown('</div>', unsafe_allow_html=True)

elif page == "Recommendations" and st.session_state.data_loaded:
    st.markdown('<h2 class="section-header">Personalized Recommendations</h2>', unsafe_allow_html=True)
    
    if st.session_state.selected_user is not None:
        # Get user profile
        data_processor = st.session_state.data_processor
        user_profile = data_processor.get_user_profile(st.session_state.selected_user)
        
        if user_profile is not None:
            # Display user profile
            display_user_profile(user_profile, data_processor.interactions_df, data_processor.content_df)
            
            # Recommendation methods
            rec_method = st.radio(
                "Recommendation Method",
                ["Collaborative Filtering", "Content-Based Filtering", "Hybrid Recommendations"]
            )
            
            # Number of recommendations to show
            n_recommendations = st.slider("Number of Recommendations", 3, 10, 5)
            
            # Option to exclude already viewed content
            exclude_watched = st.checkbox("Exclude Already Viewed Content", value=True)
            
            # Generate recommendations based on selected method
            if rec_method == "Collaborative Filtering":
                recommended_ids = st.session_state.collaborative_model.recommend_for_user(
                    st.session_state.selected_user, n_recommendations, exclude_watched
                )
                
                st.markdown('<h3 class="section-header">Collaborative Filtering Recommendations</h3>', unsafe_allow_html=True)
                st.markdown("""
                These recommendations are based on similar users' preferences. The system identifies users with similar 
                taste and recommends content they enjoyed.
                """)
                
            elif rec_method == "Content-Based Filtering":
                content_model = st.session_state.content_model
                user_profiles = content_model.create_user_profiles()
                recommended_ids = content_model.recommend_for_user(
                    st.session_state.selected_user, user_profiles, n_recommendations, exclude_watched
                )
                
                st.markdown('<h3 class="section-header">Content-Based Recommendations</h3>', unsafe_allow_html=True)
                st.markdown("""
                These recommendations are based on the content features you've previously interacted with. The system 
                analyzes content characteristics and recommends similar items.
                """)
                
            else:  # Hybrid
                # Get recommendations from both methods
                collab_rec_ids = st.session_state.collaborative_model.recommend_for_user(
                    st.session_state.selected_user, n_recommendations//2 + 1, exclude_watched
                )
                
                content_model = st.session_state.content_model
                user_profiles = content_model.create_user_profiles()
                content_rec_ids = content_model.recommend_for_user(
                    st.session_state.selected_user, user_profiles, n_recommendations//2 + 1, exclude_watched
                )
                
                # Combine recommendations
                recommended_ids = []
                for i in range(max(len(collab_rec_ids), len(content_rec_ids))):
                    if i < len(collab_rec_ids):
                        recommended_ids.append(collab_rec_ids[i])
                    if i < len(content_rec_ids):
                        recommended_ids.append(content_rec_ids[i])
                
                # Remove duplicates and limit to n_recommendations
                recommended_ids = list(dict.fromkeys(recommended_ids))[:n_recommendations]
                
                st.markdown('<h3 class="section-header">Hybrid Recommendations</h3>', unsafe_allow_html=True)
                st.markdown("""
                These recommendations combine collaborative and content-based filtering approaches to provide a diverse set
                of suggestions that leverage both user similarities and content features.
                """)
            
            # Display recommendations
            recommended_content = data_processor.get_content_details(recommended_ids)
            
            if len(recommended_content) > 0:
                for _, content_item in recommended_content.iterrows():
                    st.markdown('<div class="content-card">', unsafe_allow_html=True)
                    display_content_item(content_item, is_recommended=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No recommendations found. Try adjusting your filters or interacting with more content.")
        else:
            st.error(f"User profile not found for user ID {st.session_state.selected_user}")
    else:
        st.info("Please select a user from the sidebar to see personalized recommendations.")

elif page == "Browse Content" and st.session_state.data_loaded:
    st.markdown('<h2 class="section-header">Browse Content</h2>', unsafe_allow_html=True)
    
    # Get content data
    data_processor = st.session_state.data_processor
    content_df = data_processor.content_df
    
    # Content filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        content_type = st.selectbox("Content Type", ["All"] + list(content_df['type'].unique()))
    
    with col2:
        categories = ["All"] + list(content_df['category'].unique())
        content_category = st.selectbox("Category", categories)
    
    with col3:
        sort_by = st.selectbox("Sort By", ["Popularity", "Most Recent", "A-Z"])
    
    # Apply filters
    filtered_df = content_df.copy()
    
    if content_type != "All":
        filtered_df = filtered_df[filtered_df['type'] == content_type]
    
    if content_category != "All":
        filtered_df = filtered_df[filtered_df['category'] == content_category]
    
    # Apply sorting
    if sort_by == "Popularity":
        filtered_df = filtered_df.sort_values('popularity_score', ascending=False)
    elif sort_by == "Most Recent":
        filtered_df = filtered_df.sort_values('publish_date', ascending=False)
    else:  # A-Z
        filtered_df = filtered_df.sort_values('title')
    
    # Display filtered content
    if len(filtered_df) > 0:
        st.markdown(f"### Showing {len(filtered_df)} results")
        
        for _, content_item in filtered_df.iterrows():
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            display_content_item(content_item)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No content found with the selected filters.")

elif page == "Analytics" and st.session_state.data_loaded:
    st.markdown('<h2 class="section-header">Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    # Get data for analytics
    data_processor = st.session_state.data_processor
    users_df = data_processor.users_df
    content_df = data_processor.content_df
    interactions_df = data_processor.interactions_df
    
    # Overview statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value">{len(users_df)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Total Users</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value">{len(content_df)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Content Items</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value">{len(interactions_df)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Total Interactions</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        avg_rating = interactions_df['rating'].mean()
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-value">{avg_rating:.2f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="stat-label">Avg. Rating</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Content type distribution
    st.markdown('<h3 class="section-header">Content Type Distribution</h3>', unsafe_allow_html=True)
    type_counts = content_df['type'].value_counts()
    
    type_data = pd.DataFrame({
        'Type': type_counts.index,
        'Count': type_counts.values
    })
    
    st.bar_chart(type_data.set_index('Type'))
    
    # Popular categories
    st.markdown('<h3 class="section-header">Popular Categories</h3>', unsafe_allow_html=True)
    category_counts = content_df['category'].value_counts().head(10)
    
    category_data = pd.DataFrame({
        'Category': category_counts.index,
        'Count': category_counts.values
    })
    
    st.bar_chart(category_data.set_index('Category'))
    
    # User interaction statistics
    st.markdown('<h3 class="section-header">User Interaction Statistics</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Interactions per user
        user_interaction_counts = interactions_df['user_id'].value_counts().reset_index()
        user_interaction_counts.columns = ['User ID', 'Interaction Count']
        
        st.subheader("Interactions per User")
        st.dataframe(user_interaction_counts)
    
    with col2:
        # Most interacted content
        content_interaction_counts = interactions_df['content_id'].value_counts().head(10).reset_index()
        content_interaction_counts.columns = ['Content ID', 'Interaction Count']
        
        st.subheader("Most Popular Content")
        st.dataframe(content_interaction_counts)
    
    # Average rating by content type
    st.markdown('<h3 class="section-header">Average Rating by Content Type</h3>', unsafe_allow_html=True)
    
    # Merge interactions with content to get content type
    merged_df = pd.merge(interactions_df, content_df[['content_id', 'type']], on='content_id')
    
    # Calculate average rating by type
    avg_rating_by_type = merged_df.groupby('type')['rating'].mean().reset_index()
    
    st.bar_chart(avg_rating_by_type.set_index('type'))

else:
    # Message for when data is not loaded
    if not st.session_state.data_loaded:
        st.info("Please load the data first using the 'Load Data' button in the sidebar.")

# Footer
st.markdown("""
---
📱 **Smart Content Recommendation System** | Developed with Streamlit
""")