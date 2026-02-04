import os
import json
import random
import pickle
import faiss
import numpy as np
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
DATASET_PREFIX = "reddit"
META_DIR = "output/meta"
COMMENTS_DIR = "output/comments"
OUTPUT_FILE = f"doccano_{DATASET_PREFIX}_closest_clip_match_by_comment.jsonl"
SWAP_LOG_FILE = f"groundtruth_{DATASET_PREFIX}_closest_clip_match_by_comment.json"
GAMMA_DOMAIN = "http://gammaweb09.medien.uni-weimar.de:8080"
SEED = 42

IMAGE_INDEX_FILE = "faiss_image_index.index"
IMAGE_EMBEDDINGS_FILE = "image_embeddings.pkl"
TEXT_EMBEDDINGS_FILE = "text_embeddings.pkl"

random.seed(SEED)

# -----------------------------
# LOAD EMBEDDINGS
# -----------------------------
with open(IMAGE_EMBEDDINGS_FILE, "rb") as f:
    image_embeddings_dict = pickle.load(f)
image_ids = list(image_embeddings_dict.keys())
image_embeddings_array = np.stack(list(image_embeddings_dict.values())).astype("float32")

with open(TEXT_EMBEDDINGS_FILE, "rb") as f:
    text_embeddings_dict = pickle.load(f)
text_ids = list(text_embeddings_dict.keys())
text_embeddings_array = np.stack(list(text_embeddings_dict.values())).astype("float32")

# -----------------------------
# LOAD FAISS INDICES
# -----------------------------
image_index = faiss.read_index(IMAGE_INDEX_FILE)

# Build image ID → index position mapping for FAISS results
image_id_to_idx = {id_: i for i, id_ in enumerate(image_ids)}

# -----------------------------
# GENERATE JSONL
# -----------------------------
lines = []
swap_log = {}
counter = 0

for meta_file in tqdm(os.listdir(META_DIR), desc="Processing meta files"):
    if not meta_file.endswith("_meta.json"):
        continue

    id_ = meta_file.split("_meta.json")[0]

    # Load meta
    meta_path = os.path.join(META_DIR, meta_file)
    with open(meta_path, "r") as f:
        meta_data = json.load(f)
    meta_score = meta_data.get("score", 0)

    # Load top comment
    comments_file = os.path.join(COMMENTS_DIR, f"{id_}_comments.jsonl")
    if not os.path.exists(comments_file):
        continue

    top_comment_text = None
    top_comment_score = float("-inf")
    top_comment_raw = None
    with open(comments_file, "r") as f:
        for line in f:
            comment = json.loads(line)
            score = comment.get("score", 0)
            if score > top_comment_score:
                top_comment_score = score
                top_comment_text = comment.get("body", "")
                top_comment_raw = comment.get("body", "")

    if not top_comment_text:
        continue

    # -----------------------------
    # FIND IMAGE MATCH FOR COMMENT
    # -----------------------------
    if top_comment_raw not in text_embeddings_dict or id_ not in image_embeddings_dict:
        continue

    comment_emb = text_embeddings_dict[top_comment_raw].reshape(1, -1).astype("float32")

    # Search nearest image by comment embedding
    D, I = image_index.search(comment_emb, k=2)  # get top 2 neighbors

    # Pick nearest image that is NOT the original
    nearest_img_by_comment = None
    for idx in I[0]:
        candidate_id = image_ids[idx]
        if candidate_id != id_:
            nearest_img_by_comment = candidate_id
            break

    # If all neighbors are the original (unlikely), fallback to original
    if nearest_img_by_comment is None:
        nearest_img_by_comment = id_

    # -----------------------------
    # PAIR WITH ORIGINAL IMAGE
    # -----------------------------
    id2 = nearest_img_by_comment  # this is the nearest image to the comment

    # Random swap
    swapped = False
    if random.random() < 0.5:
        id_, id2 = id2, id_
        swapped = True

    # Build line dict
    line_dict = {
        "text": top_comment_text,
        "im_url": f"{GAMMA_DOMAIN}/{DATASET_PREFIX}/{id_}_image.jpg+{DATASET_PREFIX}/{id2}_image.jpg",
        "id": counter,
        "meta_score": meta_score
    }
    lines.append(line_dict)

    # Track swap
    swap_log[counter] = {
        "left": id_,
        "right": id2,
        "original": id2 if swapped else id_
    }

    counter += 1

# Sort by meta_score descending
lines.sort(key=lambda x: x["meta_score"], reverse=True)

# Write JSONL
with open(OUTPUT_FILE, "w") as f:
    for entry in lines:
        out_entry = {"text": entry["text"], "im_url": entry["im_url"], "id": entry["id"]}
        f.write(json.dumps(out_entry) + "\n")

# Write swap log
with open(SWAP_LOG_FILE, "w") as f:
    json.dump(swap_log, f, indent=2)

print(f"JSONL written to {OUTPUT_FILE}, swaps logged in {SWAP_LOG_FILE}")
