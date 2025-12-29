import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

# --- CONFIGURATION & PATHS ---
# Since your files are in the same folder as app.py, we use "."
DATA_PATH = "."
IMAGES_PATH = os.path.join(DATA_PATH, "images")
csv_path = os.path.join(DATA_PATH, "articles.csv")
embeddings_path = os.path.join(DATA_PATH, "embeddings.npy")
ids_path = os.path.join(DATA_PATH, "ids.npy")

# Page Config
st.set_page_config(page_title="AI Fashion Stylist", layout="wide")
st.title("🛍️ Visual Style Recommender")
st.markdown("""
**How it works:** 1. Upload an image of a clothing item.
2. The AI (CLIP) converts it into a mathematical vector.
3. It finds the nearest matching styles from our H&M inventory.
""")

# --- 1. LOAD RESOURCES (Cached for speed) ---
@st.cache_resource
def load_model():
    # Load CLIP model to CPU (Processing 1 image is fast on CPU)
    print("Loading Model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

@st.cache_data
def load_data():
    print("Loading Database...")
    # Load the "Brain"
    emb = np.load(embeddings_path)
    ids = np.load(ids_path, allow_pickle=True) # allow_pickle needed for strings
    # Load Metadata (force string type for IDs to avoid leading-zero errors)
    df = pd.read_csv(csv_path, dtype={'article_id': str})
    return emb, ids, df

# Load everything on startup
with st.spinner("Loading AI Brain... (This happens only once)"):
    model, processor = load_model()
    embeddings, item_ids, df = load_data()

# --- 2. THE SEARCH ENGINE ---
def find_similar_items(image, top_k=5):
    # Process the uploaded image
    inputs = processor(images=image, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        # Get vector for uploaded image
        query_vector = model.get_image_features(**inputs)
        # Normalize
        query_vector = query_vector / query_vector.norm(p=2, dim=-1, keepdim=True)
        query_vector = query_vector.numpy()

    # Calculate Similarity (Dot Product)
    # Shape: (1, 512) @ (512, N) -> (1, N)
    scores = np.dot(query_vector, embeddings.T).flatten()
    
    # Get Top K indices (highest scores)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    
    return top_indices, scores

# --- 3. HELPER: GET IMAGE PATH ---
def get_image_path(article_id):
    # H&M structure is usually: images/010/0108775015.jpg
    # But since you might have unzipped differently, let's check robustly
    
    # Standard H&M logic (subfolders based on first 3 digits)
    subfolder = article_id[:3]
    path_with_subfolder = os.path.join(IMAGES_PATH, subfolder, article_id + ".jpg")
    
    # Fallback: maybe images are just flat in the folder?
    path_flat = os.path.join(IMAGES_PATH, article_id + ".jpg")
    
    if os.path.exists(path_with_subfolder):
        return path_with_subfolder
    elif os.path.exists(path_flat):
        return path_flat
    else:
        return None

# --- 4. USER INTERFACE ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Upload Photo")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display User Image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Your Query Item", use_column_width=True)
        
        # Search Button
        if st.button("🔍 Find Similar Styles"):
            with st.spinner("Analyzing style patterns..."):
                indices, scores = find_similar_items(image)
                st.session_state['results'] = (indices, scores)

# --- 5. RESULTS DISPLAY ---
with col2:
    st.subheader("2. Recommendations")
    
    if 'results' in st.session_state:
        indices, scores = st.session_state['results']
        
        # Create a grid of 5 columns
        cols = st.columns(5)
        
        for i, idx in enumerate(indices):
            rec_id = item_ids[idx]
            score = scores[idx]
            
            # Get Metadata
            # Filter the dataframe safely
            meta_rows = df[df['article_id'] == rec_id]
            
            if not meta_rows.empty:
                meta = meta_rows.iloc[0]
                name = meta.get('prod_name', 'Unknown')
                category = meta.get('product_type_name', 'Unknown')
            else:
                name = "Unknown Item"
                category = "Unknown"
            
            # Show in Grid
            with cols[i]:
                img_path = get_image_path(rec_id)
                
                if img_path:
                    st.image(img_path, use_column_width=True)
                else:
                    st.warning("Img Missing")
                
                st.caption(f"**{name}**\n\n*{category}*\n\nMatch: {score:.2f}")

    else:
        st.info("Upload an image to see recommendations here.")