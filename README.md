# VLLM

This project implements a Visual Language Model (VLM)-powered multi-modal search engine using OpenAI’s CLIP and the COCO dataset. It enables users to retrieve semantically relevant images based on either a text prompt (e.g., “A Panda”) or an example image. The system uses CLIP to embed both images and text into a shared latent space and leverages FAISS for efficient similarity-based indexing and retrieval. Results are presented through an intuitive Gradio web interface, offering an interactive and scalable solution for content-based image search.


## Features
- Text-based and image-based image search
- Local dataset (COCO) visual search using FAISS
- Web search using Unsplash API
- Semantic similarity using CLIP & Evaluation metrics
- Responsive, clean Streamlit interface

## Technology Stack
- [CLIP (OpenAI)](https://github.com/openai/CLIP)
- [FAISS (Facebook)](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)
- [Unsplash API](https://unsplash.com/developers)
- [scikit-learn](https://scikit-learn.org/)

## User Interface (Streamlit)

![Image_Interface](https://drive.google.com/uc?id=1omZQGqUatfG0jQvByO1nImdHDqeFH58t)

### Application with Accuracy: 

![Image_Query](https://drive.google.com/uc?id=1ZDiqLhKA9dgzRTBv2mdtCnwLWknRT-9D)

### Demo

<video width="640" height="360" controls>
  <source src="https://github.com/karthiksuki/VLLM/blob/main/VLLM_PROJECT_DEMO.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Evaluation Metrics

- Precision by K calls – Relevance among top-K results

- Recall by K calls – Coverage of relevant images in top-K

- mAP – Mean average precision over the returned set

where, K - represent the no of images (K = 6 default)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-repo/clip-visual-search.git
cd clip-visual-search
```

### 2. Install dependencies
Make sure you have Python 3.10+ installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Prepare local dataset
- Download the [COCO dataset](https://cocodataset.org/#download)
- Download the following files:
  - `train2017` images (around 18GB).
  - `captions_train2017.json` (annotations for the train set).


### 4. Generate FAISS, captions
- `faiss_clip_index.idx` – FAISS index for CLIP image embeddings
- `captions.npy` – NumPy array of image captions

### 6. Set your Unsplash API key
Go to [Unsplash Developers](https://unsplash.com/developers), create an app, and get your **Access Key**.

Replace the following line in `vllm.py` with your key:
```python
UNSPLASH_ACCESS_KEY = "YOUR_UNSPLASH_ACCESS_KEY"
```

---

You're now ready to launch the app with:

```bash
streamlit run vllm.py
```
