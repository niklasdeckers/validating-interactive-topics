from PIL import Image
import imagehash
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

IMAGE_DIR = Path("output/images")
HASH_FN = imagehash.phash  # phash is most robust to scaling
THRESHOLD = 5  # max Hamming distance for "duplicate"

hashes = {}

for img_path in tqdm(IMAGE_DIR.glob("*")):
    try:
        with Image.open(img_path) as img:
            h = HASH_FN(img)
            hashes[img_path] = h
    except Exception as e:
        print(f"Skipping {img_path}: {e}")


from collections import defaultdict

buckets = defaultdict(list)
for path, h in tqdm(hashes.items()):
    buckets[str(h)].append(path)

for h, imgs in tqdm(buckets.items()):
    if len(imgs) > 1:
        print("Exact perceptual duplicates:", imgs)

collision_groups = [
    [p.stem.split("_")[0] for p in imgs]
    for imgs in buckets.values()
    if len(imgs) > 1
]



from pathlib import Path
import json
import shutil

BASE = Path("output")
DUP_DIR = BASE / "duplicates"
DUP_DIR.mkdir(exist_ok=True)

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def move_or_copy(src: Path, dst: Path, copy: bool):
    if not src.exists():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)

# collision_groups: list[list[str]]
for group in collision_groups:
    canonical_id = group[0]

    canonical_comments_path = BASE / "comments" / f"{canonical_id}_comments.jsonl"
    canonical_comments = load_jsonl(canonical_comments_path)

    for dup_id in group:
        is_canonical = dup_id == canonical_id

        files = [
            BASE / "images" / f"{dup_id}_image.jpg",
            BASE / "meta" / f"{dup_id}_meta.json",
            BASE / "comments" / f"{dup_id}_comments.jsonl",
        ]

        for src in files:
            dst = DUP_DIR / src.name
            move_or_copy(src, dst, copy=is_canonical)

        # merge comments from non-canonical duplicates
        if not is_canonical:
            dup_comments_path = DUP_DIR / f"{dup_id}_comments.jsonl"
            if dup_comments_path.exists():
                canonical_comments.extend(load_jsonl(dup_comments_path))

    # sort merged comments by score (descending)
    canonical_comments.sort(key=lambda x: x.get("score", 0), reverse=True)
    write_jsonl(canonical_comments_path, canonical_comments)