import os
import json
import random
import pickle
import faiss
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
DATASET_PREFIX = "lexica"
DATASET_PATH = "/mnt/ceph/storage/corpora/corpora-thirdparty/corpus-lexica-generated-images/data"
OUTPUT_FILE = f"doccano_{DATASET_PREFIX}_closest_clip_match_by_image.jsonl"
SWAP_LOG_FILE = f"groundtruth_{DATASET_PREFIX}_closest_clip_match_by_image.json"
GAMMA_DOMAIN = "http://gammaweb09.medien.uni-weimar.de:8080"
SEED = 42

IMAGE_INDEX_FILE = "faiss_image_index.index"
IMAGE_EMBEDDINGS_FILE = "image_embeddings.pkl"

random.seed(SEED)

# -----------------------------
# LOAD IMAGE EMBEDDINGS
# -----------------------------
with open(IMAGE_EMBEDDINGS_FILE, "rb") as f:
    image_embeddings_dict = pickle.load(f)
image_ids = list(image_embeddings_dict.keys())
image_embeddings_array = np.stack(list(image_embeddings_dict.values())).astype("float32")

# -----------------------------
# LOAD IMAGE FAISS INDEX
# -----------------------------
image_index = faiss.read_index(IMAGE_INDEX_FILE)

# -----------------------------
# FIND NEAREST IMAGE FOR EACH IMAGE
# -----------------------------
# k=2 because nearest neighbor includes self
distances, neighbors = image_index.search(image_embeddings_array, k=2)
nearest_dict = {image_ids[i]: image_ids[neighbors[i][1]] for i in range(len(image_ids))}

# -----------------------------
# BUILD JSONL
# -----------------------------
lines = []
swap_log = {}
counter = 0

dataset = load_dataset(DATASET_PATH, split='train')

for entry in tqdm(dataset, desc="Prompts"):
    id_ = entry["id"]
    prompt = entry["prompt"]
    # Get nearest image ID
    id2 = nearest_dict.get(id_, id_)

    # Random swap
    swapped = False
    if random.random() < 0.5:
        id_, id2 = id2, id_
        swapped = True

    # Build line dict
    line_dict = {
        "text": prompt or "",
        "im_url": f"{GAMMA_DOMAIN}/{DATASET_PREFIX}/{id_}.jpg+{DATASET_PREFIX}/{id2}.jpg",
        "id": counter
    }
    lines.append(line_dict)

    # Track swap
    swap_log[counter] = {
        "left": id_,
        "right": id2,
        "original": id2 if swapped else id_
    }

    counter += 1

# Sort randomly
random.seed(SEED)
random.shuffle(lines)

# Write JSONL
with open(OUTPUT_FILE, "w") as f:
    for entry in lines:
        out_entry = {"text": entry["text"], "im_url": entry["im_url"], "id": entry["id"]}
        f.write(json.dumps(out_entry) + "\n")

# Write swap log
with open(SWAP_LOG_FILE, "w") as f:
    json.dump(swap_log, f, indent=2)

print(f"JSONL written to {OUTPUT_FILE}, swaps logged in {SWAP_LOG_FILE}")