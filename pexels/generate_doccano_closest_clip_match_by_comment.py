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
DATASET_PREFIX = "pexels"
PROMPTS_JSON_PATH = "/var/tmp/deckersn/pexels/pexels-110k-768p-min-jpg/pexels-prompts-pairs.json"
IMAGE_DIR = "/var/tmp/deckersn/pexels/pexels-110k-768p-min-jpg/images"
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

with open(PROMPTS_JSON_PATH, 'r') as file:
    prompt_lines = json.load(file)

id_dict = {f.split(".")[0].split("-")[-1]: f for f in os.listdir(IMAGE_DIR)}

for line in tqdm(prompt_lines, desc="Prompts"):
    id_raw = list(line.keys())[0]
    if not id_raw in id_dict: continue
    id_ = id_dict[id_raw]
    prompt = list(line.values())[0]

    comment_emb = text_embeddings_dict[prompt].reshape(1, -1).astype("float32")

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
        "text": prompt or "",
        "im_url": f"{GAMMA_DOMAIN}/{DATASET_PREFIX}/{id_}+{DATASET_PREFIX}/{id2}",
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
