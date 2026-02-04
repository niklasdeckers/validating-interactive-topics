import json
import os
from glob import glob
from tqdm import tqdm
import random

DATASET_PREFIX = "pexels"
PROMPTS_JSON_PATH = "/var/tmp/deckersn/pexels/pexels-110k-768p-min-jpg/pexels-prompts-pairs.json"
IMAGE_DIR = "/var/tmp/deckersn/pexels/pexels-110k-768p-min-jpg/images"
OUTPUT_FILE = f"doccano_{DATASET_PREFIX}_single_image.jsonl"
GAMMA_DOMAIN = "http://gammaweb09.medien.uni-weimar.de:8080"
SEED = 42


records = []


with open(PROMPTS_JSON_PATH, 'r') as file:
    prompt_lines = json.load(file)

id_dict = {f.split(".")[0].split("-")[-1]: f for f in os.listdir(IMAGE_DIR)}


for line in tqdm(prompt_lines, desc="Prompts"):
    id_raw = list(line.keys())[0]
    if not id_raw in id_dict: continue
    id_ = id_dict[id_raw]
    prompt = list(line.values())[0]

    records.append({
        "text": prompt,
        "im_url": f"{GAMMA_DOMAIN}/{DATASET_PREFIX}/{id_}"
    })

# Sort randomly
random.seed(SEED)
random.shuffle(records)

# write to jsonl without score field
with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
    for r in records:
        out_f.write(json.dumps({"text": r["text"], "im_url": r["im_url"]}, ensure_ascii=False) + "\n")

print(f"[DONE] Wrote {len(records)} records to {OUTPUT_FILE}")