import os
import json
import random
import pickle
import faiss
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
META_DIR = "output/meta"
COMMENTS_DIR = "output/comments"
OUTPUT_FILE = "doccano_closest_clip_match_by_image.jsonl"
SWAP_LOG_FILE = "groundtruth_closest_clip_match_by_image.json"
GAMMA_DOMAIN = "http://gammaweb05:8080"
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

for meta_file in os.listdir(META_DIR):
    if not meta_file.endswith("_meta.json"):
        continue

    id_ = meta_file.split("_meta.json")[0]

    # Load meta
    meta_path = os.path.join(META_DIR, meta_file)
    with open(meta_path, "r") as f:
        meta_data = json.load(f)
    meta_score = meta_data.get("score", 0)

    # Load top comment (highest "score") for this image
    comments_file = os.path.join(COMMENTS_DIR, f"{id_}_comments.jsonl")
    if not os.path.exists(comments_file):
        continue

    top_comment_text = None
    top_comment_score = float("-inf")
    with open(comments_file, "r") as f:
        for line in f:
            comment = json.loads(line)
            score = comment.get("score", 0)
            if score > top_comment_score:
                top_comment_score = score
                top_comment_text = comment.get("body", "")

    # Get nearest image ID
    id2 = nearest_dict.get(id_, id_)

    # Random swap
    swapped = False
    if random.random() < 0.5:
        id_, id2 = id2, id_
        swapped = True

    # Build line dict
    line_dict = {
        "text": top_comment_text or "",
        "im_url": f"{GAMMA_DOMAIN}/{id_}_{id2}.jpg",
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