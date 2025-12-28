import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="StyleMatch AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS STYLING ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Header Styling */
    h1 {
        color: #FFFFFF;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        text-align: center;
        padding-bottom: 20px;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Card Styling for Results */
    div.stImage {
        border-radius: 10px;
        transition: transform 0.3s;
    }
    div.stImage:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    /* Upload Section */
    .upload-text {
        text-align: center;
        color: #FAFAFA;
        font-size: 1.2rem;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "articles.csv")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings.npy")
IDS_PATH = os.path.join(BASE_DIR, "ids.npy")

# --- 4. LOAD AI ENGINE ---
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

@st.cache_data
def load_database():
    if not os.path.exists(EMBEDDINGS_PATH):
        return None, None, None
    embeddings = np.load(EMBEDDINGS_PATH)
    ids = np.load(IDS_PATH, allow_pickle=True)
    df = pd.read_csv(CSV_PATH, dtype={'article_id': str})
    return embeddings, ids, df

# Load silently in background
model, processor = load_model()
embeddings, item_ids, df = load_database()

# --- 5. LOGIC & FUNCTIONS ---
def get_image_path(article_id):
    # Smart path finding (checks nested and flat structures)
    subfolder = article_id[:3]
    path_nested = os.path.join(BASE_DIR, "images", subfolder, article_id + ".jpg")
    path_root = os.path.join(BASE_DIR, subfolder, article_id + ".jpg")
    
    if os.path.exists(path_nested): return path_nested
    if os.path.exists(path_root): return path_root
    return None

def find_similar_items(image, top_k=5):
    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        query_vector = model.get_image_features(**inputs)
        query_vector = query_vector / query_vector.norm(p=2, dim=-1, keepdim=True)
        query_vector = query_vector.numpy()
    
    scores = np.dot(query_vector, embeddings.T).flatten()
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return top_indices, scores

# --- 6. THE UI LAYOUT ---

# Header Section
st.markdown("<h1>✨ StyleMatch AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #BBBBBB; margin-bottom: 40px;'>Visual Search Engine powered by OpenAI CLIP</p>", unsafe_allow_html=True)

# Main Container
if embeddings is None:
    st.error("⚠️ Database missing! Check repo files.")
else:
    # Sidebar for Upload
    with st.sidebar:
        st.header("📸 Start Here")
        uploaded_file = st.file_uploader("Upload an item", type=["jpg", "png", "jpeg"])
        
        st.markdown("---")
        st.markdown("### 💡 How it works")
        st.info(
            "This app doesn't use simple tags. "
            "It uses **Computer Vision** to understand "
            "texture, shape, and style to find "
            "visually similar matches."
        )

    # Main Content Area
    if uploaded_file:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Your Query")
            user_image = Image.open(uploaded_file).convert("RGB")
            st.image(user_image, use_column_width=True, caption="Analyzed Pattern")
            
            if st.button("🔍 Search Database", type="primary", use_container_width=True):
                with st.spinner("Scanning 5,000+ items..."):
                    indices, scores = find_similar_items(user_image)
                    st.session_state['results'] = (indices, scores)

        with col2:
            st.subheader("Top Recommendations")
            if 'results' in st.session_state:
                indices, scores = st.session_state['results']
                
                # Show results in a clean grid
                cols = st.columns(3) # 3 items per row
                for i, idx in enumerate(indices[:3]):
                    with cols[i]:
                        rec_id = item_ids[idx]
                        score = scores[idx]
                        meta = df[df['article_id'] == rec_id].iloc[0]
                        
                        img_path = get_image_path(rec_id)
                        if img_path:
                            st.image(img_path, use_column_width=True)
                            st.markdown(f"**{meta.get('prod_name', 'Item')}**")
                            st.caption(f"Match Score: {int(score*100)}%")
                        else:
                            st.warning("Image Missing")

                # Second row for remaining items
                cols_2 = st.columns(3)
                for i, idx in enumerate(indices[3:5]): # Next 2 items
                    with cols_2[i]:
                        rec_id = item_ids[idx]
                        score = scores[idx]
                        meta = df[df['article_id'] == rec_id].iloc[0]
                        
                        img_path = get_image_path(rec_id)
                        if img_path:
                            st.image(img_path, use_column_width=True)
                            st.markdown(f"**{meta.get('prod_name', 'Item')}**")
                            st.caption(f"Match Score: {int(score*100)}%")
            else:
                st.info("👈 Upload an image in the sidebar to begin.")
    
    else:
        # Welcome State (No image uploaded yet)
        st.markdown("<div class='upload-text'>waiting for input...</div>", unsafe_allow_html=True)
        # Display some random examples from your dataset as "Inspiration"
        st.subheader("Explore the Collection")
        example_cols = st.columns(5)
        random_indices = np.random.choice(len(item_ids), 5)
        for i, idx in enumerate(random_indices):
            with example_cols[i]:
                rec_id = item_ids[idx]
                path = get_image_path(rec_id)
                if path: st.image(path, use_column_width=True)
