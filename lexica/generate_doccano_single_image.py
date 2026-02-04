import json
import os
from glob import glob
from datasets import load_dataset
from tqdm import tqdm
import random

DATASET_PREFIX = "lexica"
DATASET_PATH = "/mnt/ceph/storage/corpora/corpora-thirdparty/corpus-lexica-generated-images/data"
OUTPUT_FILE = f"doccano_{DATASET_PREFIX}_single_image.jsonl"
GAMMA_DOMAIN = "http://gammaweb09.medien.uni-weimar.de:8080"
SEED = 42


records = []

dataset = load_dataset(DATASET_PATH, split='train')

for entry in tqdm(dataset, desc="Prompts"):
    id_ = entry["id"]
    prompt = entry["prompt"]

    records.append({
        "text": prompt,
        "im_url": f"{GAMMA_DOMAIN}/{DATASET_PREFIX}/{id_}.jpg"
    })

# Sort randomly
random.seed(SEED)
random.shuffle(records)

# write to jsonl without score field
with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
    for r in records:
        out_f.write(json.dumps({"text": r["text"], "im_url": r["im_url"]}, ensure_ascii=False) + "\n")

print(f"[DONE] Wrote {len(records)} records to {OUTPUT_FILE}")