import json
import os
from glob import glob

DATASET_PREFIX = "reddit"
META_DIR = "output/meta"
COMMENTS_DIR = "output/comments"
OUTPUT_FILE = f"doccano_{DATASET_PREFIX}_single_image.jsonl"
BASE_URL = "http://gammaweb09.medien.uni-weimar.de:8080"

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON decode error in {path} line {i}: {e}")

records = []

meta_files = glob(os.path.join(META_DIR, "*_meta.json"))
print(f"[DEBUG] Found {len(meta_files)} meta files")

for meta_path in meta_files:
    id_ = os.path.basename(meta_path).replace("_meta.json", "")
    print(f"\n[DEBUG] Processing ID={id_}")

    comments_path = os.path.join(COMMENTS_DIR, f"{id_}_comments.jsonl")
    if not os.path.exists(comments_path):
        print(f"[WARN] Missing comments file: {comments_path}")
        continue

    best_comment = None
    best_score = float("-inf")

    for entry in load_jsonl(comments_path):
        score = entry.get("score")
        if score is None:
            continue
        # make sure score is numeric
        try:
            score = float(score)
        except (ValueError, TypeError):
            continue
        if score > best_score:
            best_score = score
            best_comment = entry

    if not best_comment or not best_comment.get("body"):
        print(f"[WARN] No usable comment for ID={id_}")
        continue

    records.append({
        "text": best_comment["body"],
        "im_url": f"{BASE_URL}/{DATASET_PREFIX}/{id_}_image.jpg",
        "score": best_score  # keep score for sorting
    })
    print(f"[DEBUG] Collected record with score={best_score}")

# sort descending by score
records.sort(key=lambda x: x["score"], reverse=True)
print(f"[DEBUG] Sorted {len(records)} records by score descending")

# write to jsonl without score field
with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
    for r in records:
        out_f.write(json.dumps({"text": r["text"], "im_url": r["im_url"]}, ensure_ascii=False) + "\n")

print(f"[DONE] Wrote {len(records)} records to {OUTPUT_FILE}")