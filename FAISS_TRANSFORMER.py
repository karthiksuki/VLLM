import torch
from torchvision.datasets import CocoCaptions
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import os
import faiss
import numpy as np
import clip

# GPU or CPU check
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

transform = preprocess

class SafeCocoCaptions(CocoCaptions):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except FileNotFoundError:
            return None

# Coco dataset (118287 images)
coco_dataset = SafeCocoCaptions(
    root='D:/vlm_project/coco/images/train2017',
    annFile='D:/vlm_project/coco/annotations/annotations/captions_train2017.json',
    transform=transform
)

# Embedding and Transformer
def safe_collate(batch):
    batch = [item for item in batch if item is not None]
    return tuple(zip(*batch)) if batch else None

dataloader = DataLoader(coco_dataset, batch_size=32, shuffle=False, collate_fn=safe_collate)

image_embeddings = []
all_captions = []
image_filenames = []

image_ids = coco_dataset.ids

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Embedding images"):
        if batch is None:
            continue
        images, captions_list = batch
        images = torch.stack(images).to(device)
        embeddings = model.encode_image(images)
        image_embeddings.append(embeddings.cpu())
        all_captions.extend([cap[0] for cap in captions_list])

        # Track filenames based on current length of collected samples
        for _ in images:
            image_id = image_ids[len(image_filenames)]
            filename = f"{image_id:012d}.jpg"
            image_filenames.append(filename)

# Model Embedding
image_embeddings_np = torch.cat(image_embeddings, dim=0).numpy().astype('float32')

# FAISS index
index = faiss.IndexFlatL2(image_embeddings_np.shape[1])
index.add(image_embeddings_np)
faiss.write_index(index, "faiss_clip_index.idx")

# Model save
np.save("captions.npy", np.array(all_captions))
np.save("filenames.npy", np.array(image_filenames))

print("FAISS index built and saved.")
print(f"Total indexed images: {len(image_embeddings_np)}")

print("Sample:", image_filenames[:3], all_captions[:3])
