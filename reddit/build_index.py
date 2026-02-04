import os
import torch
import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import pickle
import json

# -----------------------------
# CONFIG
# -----------------------------
IMAGE_DIR = "output/images"
COMMENTS_DIR = "output/comments"

IMAGE_INDEX_FILE = "faiss_image_index.index"
TEXT_INDEX_FILE = "faiss_text_index.index"

IMAGE_EMBEDDINGS_FILE = "image_embeddings.pkl"
TEXT_EMBEDDINGS_FILE = "text_embeddings.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD CLIP MODEL
# -----------------------------
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# -----------------------------
# STEP 1: EMBED IMAGES
# -----------------------------
print("Embedding images...")
image_paths = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith("_image.jpg")]

image_embeddings_dict = {}  # id -> embedding
image_ids = []

for img_path in tqdm(image_paths, desc="Images"):
    id_ = os.path.basename(img_path).split("_image.jpg")[0]
    image_ids.append(id_)

    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)["pooler_output"]
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    image_embeddings_dict[id_] = embedding.cpu().numpy().flatten()  # store as 1D array

# Build FAISS index
image_embeddings_array = np.stack(list(image_embeddings_dict.values())).astype("float32")
d = image_embeddings_array.shape[1]
image_index = faiss.IndexFlatIP(d)
image_index.add(image_embeddings_array)

# Map index positions to IDs
image_id_list = list(image_embeddings_dict.keys())

# Save everything
faiss.write_index(image_index, IMAGE_INDEX_FILE)
with open(IMAGE_EMBEDDINGS_FILE, "wb") as f:
    pickle.dump(image_embeddings_dict, f)

print(f"Saved image index ({IMAGE_INDEX_FILE}) and embeddings dict ({IMAGE_EMBEDDINGS_FILE})")

# -----------------------------
# STEP 2: EMBED COMMENT TEXTS
# -----------------------------
print("Embedding comment texts...")

text_embeddings_dict = {}  # text string -> embedding

for comment_file in tqdm(os.listdir(COMMENTS_DIR), desc="Comments"):
    if not comment_file.endswith("_comments.jsonl"):
        continue

    comment_path = os.path.join(COMMENTS_DIR, comment_file)
    
    with open(comment_path, "r") as f:
        for line in f:
            comment = json.loads(line)
            body = comment.get("body", "").strip()
            if not body:
                continue

            inputs = processor(
                text=body,
                return_tensors="pt",
                padding=True,
                truncation=True,   # truncate to model max
                max_length=77
            ).to(DEVICE)
            
            with torch.no_grad():
                embedding = model.get_text_features(**inputs)["pooler_output"]
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            text_embeddings_dict[body] = embedding.cpu().numpy().flatten()

# Build FAISS index for text
if text_embeddings_dict:
    text_embeddings_array = np.stack(list(text_embeddings_dict.values())).astype("float32")
    d_text = text_embeddings_array.shape[1]
    text_index = faiss.IndexFlatIP(d_text)
    text_index.add(text_embeddings_array)
    
    # Map index positions to text
    text_id_list = list(text_embeddings_dict.keys())

    # Save
    faiss.write_index(text_index, TEXT_INDEX_FILE)
    with open(TEXT_EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(text_embeddings_dict, f)

    print(f"Saved text index ({TEXT_INDEX_FILE}) and embeddings dict ({TEXT_EMBEDDINGS_FILE})")
else:
    print("No comment texts found. Skipping text index.")
