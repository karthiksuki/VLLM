import streamlit as st
import torch
import clip
import faiss
import numpy as np
from PIL import Image
import os
import requests
from io import BytesIO
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INDEX_PATH = "faiss_clip_index.idx"
CAPTIONS_PATH = "captions.npy"
IMAGE_FOLDER = ""  # COCO DATASET
UNSPLASH_ACCESS_KEY = "" 
K = 7  # K images

# CLIP MODEL
model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()

# Load FAISS index and captions
index = faiss.read_index(INDEX_PATH)
captions = np.load(CAPTIONS_PATH, allow_pickle=True)

# Encode Functions
def encode_text(text):
    tokens = clip.tokenize([text]).to(DEVICE)
    with torch.no_grad():
        return model.encode_text(tokens).cpu().numpy().astype('float32')

def encode_image(image):
    image_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return model.encode_image(image_tensor).cpu().numpy().astype('float32')

def search_index(query_embedding, top_k=K):
    distances, indices = index.search(query_embedding, top_k)
    return indices[0], distances[0]

def format_results(indices, distances):
    results = []
    for i, idx in enumerate(indices):
        filename = f"{idx:012d}.jpg"
        path = os.path.join(IMAGE_FOLDER, filename)
        caption = captions[idx]
        if os.path.exists(path):
            results.append((path, caption, distances[i]))
    return results

def fetch_unsplash_images(query, count=K):
    url = "https://api.unsplash.com/search/photos"
    params = {"query": query, "per_page": count, "client_id": UNSPLASH_ACCESS_KEY}
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    return [(img['urls']['regular'], img.get('alt_description', 'No description')) for img in data['results']]

def evaluate_clip_similarity(query_emb, image_embs):
    sims = cosine_similarity(query_emb, image_embs)[0]
    sorted_indices = np.argsort(sims)[::-1]
    y_true = [1 if i == sorted_indices[0] else 0 for i in range(K)]
    precision = precision_score([1] + [0]*(K-1), y_true[:K])
    recall = recall_score([1] + [0]*(K-1), y_true[:K])
    ap = sum([y_true[i] * (sum(y_true[:i+1]) / (i+1)) for i in range(len(y_true))]) / max(1, sum(y_true))
    return precision, recall, ap, sorted_indices

# ------------------------- Streamlit UI ----------------------------
st.set_page_config(page_title="Visual Search Engine", layout="wide")
st.title("Visual Search Engine (VLLM)")
search_type = st.sidebar.radio("Search By", ["Text", "Image"])

query_emb = None
results = []
all_image_embs = []

if search_type == "Text":
    text_query = st.text_input("Enter your query")
    if text_query:
        query_emb = encode_text(text_query)
        indices, distances = search_index(query_emb)
        results = format_results(indices, distances)

        if len(results) < K:
            image_data = fetch_unsplash_images(text_query, count=K - len(results))
            for url, desc in image_data:
                try:
                    img = Image.open(BytesIO(requests.get(url).content))
                    emb = encode_image(img)
                    all_image_embs.append(emb[0])
                    results.append((url, desc, 0))
                except:
                    continue

elif search_type == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        query_image = Image.open(uploaded_file)
        st.image(query_image, caption="Query Image")
        query_emb = encode_image(query_image)
        indices, distances = search_index(query_emb)
        results = format_results(indices, distances)

        if len(results) < K:
            st.subheader("Fetching Similar Pattern of Images")
            # Encode uploaded image
            unsplash_imgs = fetch_unsplash_images("a photo", count=10)
            for url, desc in unsplash_imgs:
                try:
                    img = Image.open(BytesIO(requests.get(url).content))
                    emb = encode_image(img)
                    all_image_embs.append(emb[0])
                    results.append((url, desc, 0))
                except:
                    continue

if results:
    cols = st.columns(2)
    for i, (img_path_or_url, caption, dist) in enumerate(results):
        with cols[i % 2]:
            st.image(img_path_or_url, caption=caption)

    if all_image_embs and query_emb is not None:
        emb_stack = np.stack(all_image_embs)
        precision, recall, ap, _ = evaluate_clip_similarity(query_emb, emb_stack)
        st.markdown(f"""
        <div style='text-align:center; font-size:20px; background-color:#e3f2fd; padding:15px; border-radius:10px; color:#0d47a1;'>
            <b>Evaluation Metrics: </b><br>
            <b style='color:#2e7d32;'>Precision by k = {K}:</b> {precision:.2f} &nbsp;|&nbsp;
            <b style='color:#f57c00;'>Recall by k = {K}:</b> {recall:.2f} &nbsp;|&nbsp;
            <b style='color:#6a1b9a;'>mAP:</b> {ap:.2f}
        </div>
        """, unsafe_allow_html=True)

    elif query_emb is not None and search_type == "Image":
        try:
            filename = os.path.basename(uploaded_file.name)
            true_idx = int(filename.split(".")[0])
            precision, recall, ap = evaluate_clip_similarity(query_emb, [encode_image(Image.open(os.path.join(IMAGE_FOLDER, f"{i:012d}.jpg")))[0] for i in indices])
            st.markdown(f"""
            <div style='text-align:center; font-size:20px; background-color:#e3f2fd; padding:15px; border-radius:10px; color:#0d47a1;'>
                <b>Evaluation Metrics:</b><br>
                <b style='color:#2e7d32;'>Precision by k = {K}:</b> {precision:.2f} &nbsp;|&nbsp;
                <b style='color:#f57c00;'>Recall by k = {K}:</b> {recall:.2f} &nbsp;|&nbsp;
                <b style='color:#6a1b9a;'>mAP:</b> {ap:.2f}
            </div>
            """, unsafe_allow_html=True)
        except:
            st.markdown("<div style='text-align:center; color:gray; font-size:16px;'>Local evaluation unavailable (image not from dataset)</div>", unsafe_allow_html=True)

st.markdown("-------------------------")
